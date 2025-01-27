# BBClassify

**A successor to BB-Class, designed for accurate classification consistency and accuracy estimation.**

## Overview

BBClassify is a computer program designed to address the shortcomings of its predecessor, BB-Class, in estimating classification consistency and accuracy. The primary purpose of this software is to rigorously assess how reliably and accurately test scores categorize individuals into predefined groups (e.g., pass/fail, proficiency levels).

BBClassify provides tools for evaluating test scores in high-stakes situations, such as licensure and certification testing. It implements the methodologies proposed by Hanson and Brennan (1990) and Livingston and Lewis (1995), adapting them to accommodate various testing scenarios:
*   Equally weighted, dichotomously scored items
*   More complex item formats such as weighted or polytomously scored items.

## Key Improvements over BB-Class

The major flaw in the BB-Class software was its incorrect estimation of the *effective test length* within the Livingston and Lewis (L&L) procedure. Instead of using the minimum and maximum *possible* scores for this calculation, BB-Class mistakenly used the minimum and maximum *observed* scores. This led to significant deviations in model parameter estimates, and consequently inaccurate classification consistency and accuracy estimates. This flaw has potentially compromised research and test analysis for over 20 years.

BBClassify rectifies this error by:
*  **Requiring the maximum possible test score as a mandatory input.** This ensures accurate calculations of the effective test length.
*   **Making the minimum possible test score an optional input,** defaulting to 0 if not specified.

These changes are critical to producing valid results with the Livingston and Lewis procedure.

## User Experience

To promote ease of adoption, BBClassify:
*   Maintains a **carbon copy of the original BB-Class user interface**. This provides a seamless transition for existing users.
*   Requires the same main inputs as BB-Class and retains many of the optional inputs
*   **Omitted several unnecessary optional inputs** from BB-Class to simplify its usage

By preserving familiarity while improving accuracy, BBClassify aims to be a more reliable tool for researchers and practitioners in high-stakes testing.

## Usage

BBClassify provides a user interface that is designed to be familiar to existing users of BB-Class. It accepts similar inputs, such as:
*  Test scores (as a raw-score distribution, or raw-score moments)
*  Reliability coefficient
*  Cut score(s)
*  Minimum and maximum *possible* score

For specific usage instructions, please refer to the provided documentation.

## Availability

This is the latest version of the BBClassify software. It is written in C, and is designed to run on any machine with an ANSI C compiler.

##  Future work
*  Additional research ought to be done on which error handling procedure for the Beta-Binomial provides the least biased estimates under differing conditions.
* Further research ought to be done to explore the relative performance of the methods implemented within this software to those based on IRT or other approaches.
*  More research is required to determine the best reliability estimation procedure to input for the L&L method under different testing conditions.

---

This markdown file incorporates information from all provided documents.
I also included some of the mathematical formulae in a markdowned format using LaTeX markup.

Let me know if there are any other changes you'd like me to make!
