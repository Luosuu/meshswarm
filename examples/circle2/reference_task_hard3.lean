-- Proves only the weakest bound, to confirm in the self-test that a real proof is accepted.
namespace CirclePackingConstants

theorem three_upper_a : ∀ r : ℝ, Packable 3 r → r ≤ (2 - Real.sqrt 2) / 2 := by
  intro r ⟨hr0, hr1, c, hin, hsep⟩
  have h2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have h0 : 0 ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
  have h := hsep 0 1 (by decide)
  obtain ⟨a1, a2, a3, a4⟩ := hin 0
  obtain ⟨b1, b2, b3, b4⟩ := hin 1
  simp only [sqDist] at h
  have hx : ((c 0).1 - (c 1).1) ^ 2 ≤ (1 - 2 * r) ^ 2 := by
    apply sq_le_sq' <;> linarith
  have hy : ((c 0).2 - (c 1).2) ^ 2 ≤ (1 - 2 * r) ^ 2 := by
    apply sq_le_sq' <;> linarith
  have key : (2 * r) ^ 2 ≤ 2 * (1 - 2 * r) ^ 2 := by linarith
  by_contra hcon
  have hcon := not_le.mp hcon
  nlinarith [mul_nonneg hr0 h0, sq_nonneg (Real.sqrt 2 - 1)]

theorem three_upper_b : ∀ r : ℝ, Packable 3 r → r ≤ 27 / 100 := by
  sorry

theorem three_upper_c : ∀ r : ℝ, Packable 3 r → r ≤ 26 / 100 := by
  sorry

theorem three_upper_d : ∀ r : ℝ, Packable 3 r → r ≤ 255 / 1000 := by
  sorry

theorem three_upper : ∀ r : ℝ, Packable 3 r → r ≤ (Real.sqrt 6 - Real.sqrt 2) / (2 * (1 + (Real.sqrt 6 - Real.sqrt 2))) := by
  sorry

end CirclePackingConstants
