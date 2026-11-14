-- Proves the two easiest targets, to confirm in the self-test that a real proof is accepted.
namespace OAI.Paper320
open scoped Classical

theorem sensitivityAt_le_sensitivity :
    ∀ {I : Type} [Fintype I] (f : (I → Bool) → Bool) (x : I → Bool), sensitivityAt f x ≤ sensitivity f := by
  intro I _ f x
  exact Finset.le_sup (f := sensitivityAt f) (Finset.mem_univ x)

theorem blockSensitivityAt_le_blockSensitivity :
    ∀ {I : Type} [Fintype I] (f : (I → Bool) → Bool) (x : I → Bool), blockSensitivityAt f x ≤ blockSensitivity f := by
  intro I _ f x
  exact Finset.le_sup (f := blockSensitivityAt f) (Finset.mem_univ x)

theorem sensitivityAt_le_blockSensitivityAt :
    ∀ {I : Type} [Fintype I] (f : (I → Bool) → Bool) (x : I → Bool), sensitivityAt f x ≤ blockSensitivityAt f x := by
  sorry

theorem sensitivity_le_blockSensitivity :
    ∀ {I : Type} [Fintype I] (f : (I → Bool) → Bool), sensitivity f ≤ blockSensitivity f := by
  sorry

theorem le_blockSensitivityAt_of_family :
    ∀ {I K : Type} [Fintype I] [Fintype K] (f : (I → Bool) → Bool) (x : I → Bool) (blocks : K → Finset I)
    (_hne : ∀ k, (blocks k).Nonempty)
    (_hd : Pairwise fun k l => Disjoint (blocks k) (blocks l))
    (_hb : ∀ k, f (flip x (blocks k)) ≠ f x),
    Fintype.card K ≤ blockSensitivityAt f x := by
  sorry

theorem exists_blockSensitivityAt_family :
    ∀ {I : Type} [Fintype I] (f : (I → Bool) → Bool) (x : I → Bool),
    ∃ blocks : Finset (Finset I), blocks.card = blockSensitivityAt f x ∧
      (∀ A ∈ blocks, ∀ B ∈ blocks, A ≠ B → Disjoint A B) ∧
      (∀ B ∈ blocks, B.Nonempty ∧ f (flip x B) ≠ f x) := by
  sorry

theorem ballPredicate_sensitivityAt_zero_le :
    ∀ {I C : Type} [Fintype I] [Fintype C] (centers : C → I → Bool) (R : ℕ)
    (_hsep : ∀ a b, a ≠ b → 2 * (R + 1) < hammingDist (centers a) (centers b))
    (x : I → Bool) (_hx : ballPredicate centers R x = false),
    sensitivityAt (ballPredicate centers R) x ≤ R + 1 := by
  sorry

theorem initialFamily_joint :
    ∀ (d M : ℕ) (b : Bool), jointSensitivity (initialFamily d M) b = 0 := by
  sorry

theorem sensitivity_orCopies_le :
    ∀ {I K : Type} [Fintype K] [Fintype I] (f : (I → Bool) → Bool) (S₀ S₁ : ℕ)
    (_h₀ : ∀ x, f x = false → sensitivityAt f x ≤ S₀)
    (_h₁ : ∀ x, f x = true → sensitivityAt f x ≤ S₁),
    sensitivity (orCopies (K := K) f) ≤ max (Fintype.card K * S₀) S₁ := by
  sorry

theorem blockSensitivityAt_compose_zero_le :
    ∀ {I J : Type} [Fintype I] [Fintype J] (f : (I → Bool) → Bool)
    (g : (J → Bool) → Bool) (_hg : g (fun _ => false) = false),
    blockSensitivityAt f (fun _ => false) * blockSensitivityAt g (fun _ => false) ≤
      blockSensitivityAt (compose f g) (fun _ => false) := by
  sorry

theorem quantitative_profile_bounds :
    ∀ (d L M r : ℕ) (_hM : 0 < M)
    (_hbudget : ((max r 16 : ℕ) : ℝ) / M * 3 ^ (d + 1) ≤ 2)
    (S J : ℕ → Bool → ℕ)
    (_hS₀ : S 0 false ≤ L) (_hS₁ : S 0 true ≤ L * M)
    (_hJ₀ : J 0 false = 0) (_hJ₁ : J 0 true = 0)
    (_hstepS₀ : ∀ n < d, S (n + 1) false ≤ 16 * S n false + S n true + 3 * J n true)
    (_hstepS₁ : ∀ n < d, S (n + 1) true ≤ M ^ 2 * S n false + r * S n true)
    (_hstepJ₀ : ∀ n < d, J (n + 1) false ≤ 16 * S n false + 3 * J n true)
    (_hstepJ₁ : ∀ n < d, J (n + 1) true ≤ M ^ 2 * J n false + r * S n true),
    ∀ n ≤ d, S n false ≤ 2 * L * M ^ n ∧ S n true ≤ 2 * L * M ^ (n + 1) := by
  sorry

theorem baseline_arithmetic :
    let M : ℕ := 9 ^ 10
    ((2 * (9 + 2) * M ^ 10 : ℕ) : ℝ) ^ (2.000252407 : ℝ) ≤
      ((M ^ 2 * (2 * M ^ 2 + 1) ^ 9 : ℕ) : ℝ) := by
  sorry

end OAI.Paper320
