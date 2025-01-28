# BBClassify

BBClassify is program for the psychometric analysis of classification accuracy and consistency based on test scores using the beta-binomial model. It is intended to succeed BB-Class ([Brennan, 2004](https://brennancrickgenova.org/classification-decision-consistency-programs/)) which contains a significant error that appears to have gone by unnoticed for over 20 years. This is of significant concern seen as how BB-Class has seen widespread use not only in research but also for the purposes of quality assurance in licensure and certification testing.

## About BB-Class

The **BB-Class** software calculates classification consistency and accuracy using methodologies from the Beta-Binomial model and extensions, including:
- **Hanson and Brennan ([1990](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1745-3984.1990.tb00753.x))**: For equally weighted, dichotomously scored items.
- **Livingston and Lewis (1995)**: For complex scenarios (e.g., weighted or polytomously scored items).

### Critical Issue in BB-Class
The Livingston & Lewis (L&L) procedure in BB-Class incorrectly estimates the **effective test length** parameter.<br><br>
The equation for calculating the effective test length is:

<div align="center">
<picture>
   <source srcset="https://quicklatex.com/cache3/ab/ql_f6ece2871c700a2a20c15f863bd48dab_l3.png" media="(prefers-color-scheme: light)">
   <source srcset="https://quicklatex.com/cache3/bc/ql_1f22e5b3830a792b9090b035feb729bc_l3.png" media="(prefers-color-scheme: dark)">
   <img src="https://quicklatex.com/cache3/ab/ql_f6ece2871c700a2a20c15f863bd48dab_l3.png" alt="Default Image">
</picture>
</div>
<br>
Where μ is the mean of the observed-score distribution, σ² is the variance of the observed-score distribution, and ρ is the test-score reliability coefficient. Xₘᵢₙ and Xₘₐₓ refer to the <i><b>theoretically possible</b></i> minimum and maximum test-scores.<br><br>
 
The error in BB-Class is to use the <i><b>empirically observed</b></i> minimum and maximum test scores for Xₘᵢₙ and Xₘₐₓ. That is, as long as it is possible to achieve a test-score of 0 by either not answering or answering all questions incorrectly, Xₘᵢₙ is 0 even if the lowest score actually observed in the sample is, for example, 10. Likewise, if it is theoretically possible to achieve a test score of 100 (that is, if answering all questions correctly would yield a test score of 100) then Xₘₐₓ is 100, even if the highest score actually observed in the sample is 90. 

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
- **Backward Compatibility**: Maintains input requirements (with one necessary exception required to fix the L&L error).  

### Design principles
The design of BBClassify prioritizes performance without sacrificing clarity through documentation. Recognizing Python’s reputation as a relatively "slow" programming language, every effort has been made to optimize performance, ensuring that BBClassify can handle the intensive demands of simulation studies where execution time can quickly scale from seconds to hours. To achieve this, code readability is deprioritized in favor of extensive commenting and thorough documentation, enabling developers and users to understand the implementation while maintaining high efficiency.

At the same time, user experience is at the forefront of BBClassify’s design. The interface has been developed with user-friendliness and responsiveness as primary goals. Intuitive workflows and clear, helpful error messages guide users through resolving potential issues, ensuring a smooth experience. The interface is designed to remain responsive at all times, so users never feel uncertain about the program’s status or suspect it has crashed.

Finally, BBClassify embraces the principles of open source development, fostering collaboration, transparency, and trust. By making the code publicly available, the program invites contributions for bug-fixing and future enhancements while allowing users to inspect the implementation of statistical methods to ensure they are scientifically sound and aligned with best practices. This open approach ensures that BBClassify can evolve with the community’s needs while maintaining its reliability and credibility.

---

For support or contributions, please refer to the project documentation or contact the development team.
