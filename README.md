# BBClassify

BBClassify is the successor to BB-Class, offering corrected and enhanced methodologies for estimating classification consistency and accuracy in high-stakes testing.

---

## Overview

BBClassify is a software tool designed to calculate **classification consistency** and **accuracy** using the Beta-Binomial model and its extensions. It implements methodologies from:
- **Hanson and Brennan (1990)**
- **Livingston and Lewis (1995)**

The software supports both simple scenarios (e.g., dichotomously scored items) and complex cases (e.g., weighted or polytomously scored items).

---

## The Problem with BB-Class

The original BB-Class software contained a critical flaw in its implementation of the **Livingston and Lewis (L&L) procedure**, leading to inaccurate estimates:

### Key Issue: Effective Test Length Miscalculation
- **Error**: BB-Class used *observed* score ranges (min/max observed scores) instead of *possible* score ranges (min/max possible scores) to estimate the effective test length.
- **Impact**: 
  - Example: For a test with a possible score range of **0–100** and observed scores **29–92**, BB-Class incorrectly estimated an effective test length of **40** instead of the true value of **100**.
  - This error compromised classification consistency/accuracy estimates, affecting high-stakes decisions in licensure, certification, and research for **over 20 years**.

---

## How BBClassify Solves This

BBClassify addresses BB-Class's limitations while preserving familiarity for users:

### Enhancements
- ✅ **Correct Effective Test Length Calculation**  
  - Requires **maximum possible test score** as mandatory input.
  - Optional **minimum possible test score** (defaults to `0` if unspecified).
- 🛠️ **Simplified Inputs**  
  - Removes redundant optional inputs from BB-Class.
- 🔄 **Seamless Transition**  
  - Retains the original BB-Class user interface for ease of adoption.

---

## Why It Matters

- **Reliability**: Ensures accurate parameter estimates for classification consistency/accuracy.
- **High-Stakes Applications**: Suitable for licensure, certification, and other critical testing scenarios.
- **Research Integrity**: Corrects decades of flawed outputs, preventing future misinformed decisions.

---

## Transitioning from BB-Class

BBClassify is designed for effortless adoption:
- Identical workflow and input requirements (with minor simplifications).
- No learning curve for existing BB-Class users.

---

**BBClassify** empowers researchers and practitioners to trust their results, combining decades of methodological rigor with critical corrections for modern testing needs.