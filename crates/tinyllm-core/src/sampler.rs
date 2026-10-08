//! Token sampling: argmax (temperature 0) or multinomial with temperature.
//! xorshift64* RNG (the same one llama2.c uses) to avoid depending on `rand` in no_std.

pub struct Sampler {
    pub temperature: f32,
    rng: u64,
}

impl Sampler {
    pub fn new(temperature: f32, seed: u64) -> Self {
        Sampler { temperature, rng: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed } }
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        // The multiplication spreads the bits: without it, small seeds produce
        // numbers close to 0 on the first calls and the draw always lands
        // on token 0 (<unk>).
        let x = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        // 24 mantissa bits -> [0, 1)
        (x >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Picks the next token. Modifies `logits` in place (they become probabilities).
    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        let inv_t = 1.0 / self.temperature;
        let mut max = f32::NEG_INFINITY;
        for v in logits.iter_mut() {
            *v *= inv_t;
            if *v > max {
                max = *v;
            }
        }
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = libm::expf(*v - max);
            sum += *v;
        }
        let r = self.next_f32() * sum;
        let mut cdf = 0.0f32;
        for (i, &p) in logits.iter().enumerate() {
            cdf += p;
            if r < cdf {
                return i;
            }
        }
        logits.len() - 1
    }
}

pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}
