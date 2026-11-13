-- A hand-written solution, used only by `run.py --selftest` to confirm that the four
-- statements are provable as stated. It is never shown to the agents.
namespace CirclePackingConstants

theorem sqrt_two_sq : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)

theorem two_packable : Packable 2 ((2 - Real.sqrt 2) / 2) := by
  have h2 : Real.sqrt 2 ^ 2 = 2 := sqrt_two_sq
  have h0 : 0 ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
  have h1 : Real.sqrt 2 ≤ 2 := by nlinarith
  have h3 : 1 ≤ Real.sqrt 2 := by nlinarith
  refine ⟨by linarith, by linarith, ?_⟩
  refine ⟨![((2 - Real.sqrt 2) / 2, (2 - Real.sqrt 2) / 2),
            (1 - (2 - Real.sqrt 2) / 2, 1 - (2 - Real.sqrt 2) / 2)], ?_, ?_⟩
  · intro i
    fin_cases i <;> simp [InInnerSquare] <;> (repeat' constructor) <;> linarith
  · intro i j hij
    fin_cases i <;> fin_cases j <;> simp [sqDist] at hij ⊢ <;> nlinarith

theorem two_upper : ∀ r : ℝ, Packable 2 r → r ≤ (2 - Real.sqrt 2) / 2 := by
  intro r ⟨hr0, hr1, c, hin, hsep⟩
  have h2 : Real.sqrt 2 ^ 2 = 2 := sqrt_two_sq
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

theorem r_two : r_n 2 = (2 - Real.sqrt 2) / 2 := by
  unfold r_n
  apply le_antisymm
  · exact csSup_le ⟨_, two_packable⟩ (fun r hr => two_upper r hr)
  · exact le_csSup ⟨_, fun r hr => two_upper r hr⟩ two_packable

theorem c_n_two : c_n 2 = Real.pi * (3 - 2 * Real.sqrt 2) := by
  have h2 : Real.sqrt 2 ^ 2 = 2 := sqrt_two_sq
  unfold c_n
  rw [r_two]
  push_cast
  ring_nf
  rw [h2]
  ring

end CirclePackingConstants
