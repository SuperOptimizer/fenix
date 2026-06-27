// core/optimize.hpp — first-party AdamW optimizer over a flat parameter vector. The engine
// for the diffeomorphic fit (flow + gap-expander + affine + dr) and other gradient-based
// solves. Decoupled from any loss: caller supplies gradients. See winding/CLAUDE.md.
#pragma once

#include "core/types.hpp"

#include <cmath>
#include <span>
#include <vector>

namespace fenix {

struct AdamWConfig {
    f32 lr = 1e-2f;
    f32 beta1 = 0.9f;
    f32 beta2 = 0.999f;
    f32 eps = 1e-8f;
    f32 weight_decay = 0.0f;  // decoupled (AdamW)
};

class AdamW {
public:
    explicit AdamW(usize n, AdamWConfig cfg = {})
        : cfg_(cfg), m_(n, 0.0f), v_(n, 0.0f) {}

    // One step: params -= lr * (mhat / (sqrt(vhat)+eps) + wd*params).
    void step(std::span<f32> params, std::span<const f32> grad) {
        ++t_;
        const f32 b1t = 1.0f - std::pow(cfg_.beta1, static_cast<f32>(t_));
        const f32 b2t = 1.0f - std::pow(cfg_.beta2, static_cast<f32>(t_));
        for (usize i = 0; i < params.size(); ++i) {
            const f32 g = grad[i];
            m_[i] = cfg_.beta1 * m_[i] + (1.0f - cfg_.beta1) * g;
            v_[i] = cfg_.beta2 * v_[i] + (1.0f - cfg_.beta2) * g * g;
            const f32 mhat = m_[i] / b1t;
            const f32 vhat = v_[i] / b2t;
            params[i] -= cfg_.lr * (mhat / (std::sqrt(vhat) + cfg_.eps) + cfg_.weight_decay * params[i]);
        }
    }

    void set_lr(f32 lr) { cfg_.lr = lr; }
    [[nodiscard]] int t() const { return t_; }

private:
    AdamWConfig cfg_;
    std::vector<f32> m_, v_;
    int t_ = 0;
};

// Convenience: minimize a loss via AdamW with a caller-provided gradient function for
// `iters` steps. grad_fn(params, out_grad) fills out_grad; returns nothing.
template <class GradFn>
void minimize(std::span<f32> params, GradFn&& grad_fn, int iters, AdamWConfig cfg = {}) {
    AdamW opt(params.size(), cfg);
    std::vector<f32> g(params.size());
    for (int it = 0; it < iters; ++it) {
        grad_fn(std::span<const f32>(params), std::span<f32>(g));
        opt.step(params, g);
    }
}

}  // namespace fenix
