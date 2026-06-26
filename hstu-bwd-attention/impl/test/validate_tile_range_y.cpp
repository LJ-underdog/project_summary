// SPDX-License-Identifier: MIT
// HSTU bwd M2 — offline validator for GetTileRangeAlongY (DESIGN §5.4, M2 hard prereq).
// M8/B2 extension: now also exercises the *tightened* causal NoLocal range (self + CROSS,
// incl. diff_q_kv_len) and sweeps seqlen_q != seqlen_k.
//
// Asserts, for every (seqlen_q, seqlen_k, 5-factor config) x every KV tile [n0, n0+kN0):
//     [y_start, y_end) ⊇ { sq : ∃ sk ∈ tile, IsTokenPairInsideMask(sq, sk) }
// i.e. the Y-range returned by the mask is a SUPERSET of the true attending Q-row set
// (宁多算不漏). Correctness of the GPU mask path depends on this holding. This is the
// hardest under-tighten gate (exhaustive, no GPU needed).
//
// Build:  hipcc -std=c++17 -I<18_hstu_attention dir> -I<ck_hstu include> validate_tile_range_y.cpp -o validate_tile_range_y
// Run  :  ./validate_tile_range_y         (exit 0 = all green)

#include <cstdio>
#include <vector>
#include <tuple>

#include <ck_tile/core.hpp> // ck_tile::make_tuple / number (masking header relies on includer)
#include "hstu_block_masking.hpp"

using namespace ck_tile;

// bwd hd64 tiles (kM0 = YTile = Q tile, kN0 = XTile = KV tile). The tightening logic is
// tile-size agnostic; a second (smaller) tile config is also swept to stress alignment.
static long g_checks = 0;
static long g_fail   = 0;

template <int kM0, int kN0, typename MakeMaskFn>
void check_config(const char* tag,
                  int seqlen_q,
                  int seqlen_k,
                  int contextual,
                  int num_target,
                  int window,
                  int min_full,
                  MakeMaskFn make_mask)
{
    // iterate KV tiles over the K dimension
    for(int n0 = 0; n0 < seqlen_k; n0 += kN0)
    {
        auto mask = make_mask();
        const auto yr = mask.GetTileRangeAlongY(n0, number<kM0>{}, number<kN0>{});
        const int y_start = yr.at(number<0>{});
        const int y_end   = yr.at(number<1>{});

        const int col_lo = n0;
        const int col_hi = (n0 + kN0 < seqlen_k) ? (n0 + kN0) : seqlen_k;

        for(int sq = 0; sq < seqlen_q; ++sq)
        {
            bool attends = false;
            for(int sk = col_lo; sk < col_hi; ++sk)
            {
                auto m2 = make_mask(); // IsTokenPairInsideMask is non-const; fresh copy is cheap
                if(m2.IsTokenPairInsideMask(sq, sk))
                {
                    attends = true;
                    break;
                }
            }
            if(attends)
            {
                ++g_checks;
                if(!(y_start <= sq && sq < y_end))
                {
                    ++g_fail;
                    if(g_fail <= 40)
                        printf("  FAIL [%s tile=%dx%d] sq=%d sk_q=%d sk_k=%d ctx=%d ntgt=%d win=%d "
                               "mf=%d : KVtile@%d attends sq=%d but range=[%d,%d)\n",
                               tag, kM0, kN0, seqlen_q, seqlen_q, seqlen_k, contextual, num_target,
                               window, min_full, n0, sq, y_start, y_end);
                }
            }
        }
    }
}

// Run a config under several tile shapes so the kM0-alignment is stressed (hd64: 32x128;
// hd256-ish: 16x64). Superset must hold for every tile shape.
template <bool kUseCausal, typename MakeMaskFn>
void check_all_tiles(const char* tag, int sq, int sk, int ctx, int ntgt, int win, int mf,
                     MakeMaskFn mk)
{
    check_config<32, 128>(tag, sq, sk, ctx, ntgt, win, mf, mk);
    check_config<16, 64>(tag, sq, sk, ctx, ntgt, win, mf, mk);
}

template <bool kUseCausal>
void sweep_self()
{
    const int seqlens[]  = {64, 128, 130, 200, 256, 384, 512};
    const int ctxs[]     = {0, 6};
    const int ntgts[]    = {0, 8, 32};
    const int windows[]  = {0, 5, 16, 64};
    const int minfulls[] = {0, 6, 64};

    for(int seqlen : seqlens)
        for(int ctx : ctxs)
            for(int ntgt : ntgts)
            {
                if(ntgt >= seqlen)
                    continue;
                for(int win : windows)
                    for(int mf : minfulls)
                    {
                        const int eff_mf = (seqlen - ntgt > mf) ? mf : (seqlen - ntgt);
                        if(win > 0)
                        {
                            using MaskT = typename HstuBlockMasking<false, kUseCausal, true>::Type;
                            check_all_tiles<kUseCausal>(
                                kUseCausal ? "self,causal,local" : "self,nocausal,local",
                                seqlen, seqlen, ctx, ntgt, win, eff_mf, [&] {
                                    return make_hstu_self_attention_block_mask_with_local<MaskT>(
                                        true, seqlen, ctx, ntgt, win, eff_mf);
                                });
                        }
                        else
                        {
                            using MaskT = typename HstuBlockMasking<false, kUseCausal, false>::Type;
                            check_all_tiles<kUseCausal>(
                                kUseCausal ? "self,causal,nolocal" : "self,nocausal,nolocal",
                                seqlen, seqlen, ctx, ntgt, win, eff_mf, [&] {
                                    return make_hstu_self_attention_block_mask_without_local<MaskT>(
                                        seqlen, ctx, ntgt);
                                });
                        }
                    }
            }
}

// Cross-attention: seqlen_q != seqlen_k (both directions). target_in_kv == false, so the K
// side carries NO targets (max_k_uih_len == seqlen_k); num_target lives on the Q side
// (max_q_uih_len = seqlen_q - num_target). diff_q_kv_len = seqlen_k - max_q_uih_len.
template <bool kUseCausal>
void sweep_cross()
{
    const int seqlen_qs[] = {128, 200, 256};
    const int seqlen_ks[] = {64, 128, 256, 384};
    const int ctxs[]      = {0, 6};
    const int ntgts[]     = {0, 8, 32};
    const int windows[]   = {0, 5, 16, 64};
    const int minfulls[]  = {0, 6, 64};

    for(int sq : seqlen_qs)
        for(int sk : seqlen_ks)
            for(int ctx : ctxs)
                for(int ntgt : ntgts)
                {
                    if(ntgt >= sq)
                        continue;
                    for(int win : windows)
                        for(int mf : minfulls)
                        {
                            const int eff_mf = (sq - ntgt > mf) ? mf : (sq - ntgt);
                            if(win > 0)
                            {
                                using MaskT =
                                    typename HstuBlockMasking<true, kUseCausal, true>::Type;
                                check_all_tiles<kUseCausal>(
                                    kUseCausal ? "cross,causal,local" : "cross,nocausal,local",
                                    sq, sk, ctx, ntgt, win, eff_mf, [&] {
                                        return make_hstu_cross_attention_block_mask_with_local<
                                            MaskT>(true, sq, sk, ctx, ntgt, win, eff_mf);
                                    });
                            }
                            else
                            {
                                using MaskT =
                                    typename HstuBlockMasking<true, kUseCausal, false>::Type;
                                check_all_tiles<kUseCausal>(
                                    kUseCausal ? "cross,causal,nolocal" : "cross,nocausal,nolocal",
                                    sq, sk, ctx, ntgt, win, eff_mf, [&] {
                                        return make_hstu_cross_attention_block_mask_without_local<
                                            MaskT>(sq, sk, ctx, ntgt);
                                    });
                            }
                        }
                }
}

int main()
{
    printf("HSTU bwd GetTileRangeAlongY offline superset validator (B2+B3: tightened causal NoLocal "
           "+ local/window WithLocal, self+cross; tile shapes 32x128 & 16x64)\n");
    sweep_self<true>();   // causal
    sweep_self<false>();  // non-causal
    sweep_cross<true>();  // causal cross
    sweep_cross<false>(); // non-causal cross
    printf("checks=%ld  failures=%ld -> %s\n", g_checks, g_fail,
           g_fail == 0 ? "ALL GREEN" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
