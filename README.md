# BBClassify

BBClassify is program for the psychometric analysis of classification accuracy and consistency based on test scores using the beta-binomial model. It is intended to succeed BB-Class ([Brennan, 2004](https://brennancrickgenova.org/classification-decision-consistency-programs/)) which contains a serious error that appears to have gone by unnoticed for over 20 years.

## About BB-Class

The **BB-Class** software calculates classification consistency and accuracy using methodologies from the Beta-Binomial model and extensions, including:
- **Hanson and Brennan (1990)**: For equally weighted, dichotomously scored items.
- **Livingston and Lewis (1995)**: For complex scenarios (e.g., weighted or polytomously scored items).

### Critical Issue in BB-Class
The Livingston & Lewis (L&L) procedure in BB-Class incorrectly estimates **effective test length**:
The equation for calculating the effective test length is:

<div align="center">
<picture>
   <source srcset="https://quicklatex.com/cache3/fe/ql_9e9ed976512b02ef684fa8aa746866fe_l3.png" media="(prefers-color-scheme: light)">
   <source srcset="https://quicklatex.com/cache3/5f/ql_b021354c178e47b8da9e1d6f8ba7fa5f_l3.png" media="(prefers-color-scheme: dark)">
   <img src="https://quicklatex.com/cache3/fe/ql_9e9ed976512b02ef684fa8aa746866fe_l3.png" alt="Default Image">
</picture>
</div>

- Using the *empirically observed* minimum and maximum test scores instead of the *theoretically possible* minimum and maximum test scores.  

**This error leads to:**
- Inaccurate model parameter estimates.
- Flawed classification consistency and accuracy results.
- Compromised research and policy decisions in high-stakes testing.

## Introducing BBClassify

BBClassify resolves the core issue in BB-Class, maintains its user interface, and enhances user friendliness.

### Key Enhancements  
1. **Preserved Methodological Integrity**: Corrects the implementation of the L&L procedure.  
2. **Accurate Effective Test Length Calculation**:  
   - **New Mandatory input**: Maximum possible test score.  
   - **New Optional input**: Minimum possible test score (defaults to 0).
3. **Simplified Inputs**: Removes unnecessary optional parameters from BB-Class for streamlined usage.  
4. **User-Friendly Error Handling**:  
   - Provides **clear, actionable error messages** for invalid inputs or data inconsistencies, guiding users to resolve issues quickly.

### Seamless Transition
- **Familiar Interface**: Mirrors the BB-Class user interface for effortless adoption.  
- **Backward Compatibility**: Maintains input requirements (with minor optional simplifications).  

## Impact
BBClassify ensures researchers and practitioners can trust its outputs for critical applications, mitigating risks from past inaccuracies and fostering reliable test analysis.

---

For support or contributions, please refer to the project documentation or contact the development team.
