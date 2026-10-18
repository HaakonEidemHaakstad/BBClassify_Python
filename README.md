# BBClassify

BBClassify estimates the accuracy and consistency of test-based classifications with the beta-binomial model and its extensions: the procedures of Hanson and Brennan ([1990](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1745-3984.1990.tb00753.x)) and Livingston and Lewis ([1995](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1745-3984.1995.tb00462.x)).

It is intended to succeed BB-CLASS ([Brennan, 2004](https://brennancrickgenova.org/classification-decision-consistency-programs/)), which contains an error in its Livingston and Lewis procedure that appears to have gone unnoticed for over 20 years. BB-CLASS has been widely used, both in research and for quality assurance in licensure and certification testing. BBClassify uses the same control cards as BB-CLASS, with one addition needed to correct the error, and reproduces BB-CLASS's published results wherever BB-CLASS is not affected by the error.

## The error in BB-CLASS

The Livingston and Lewis (LL) procedure replaces the number of items with an *effective test length*:

<div align="center">
<picture>
   <source srcset="https://quicklatex.com/cache3/ab/ql_f6ece2871c700a2a20c15f863bd48dab_l3.png" media="(prefers-color-scheme: light)">
   <source srcset="https://quicklatex.com/cache3/bc/ql_1f22e5b3830a792b9090b035feb729bc_l3.png" media="(prefers-color-scheme: dark)">
   <img src="https://quicklatex.com/cache3/ab/ql_f6ece2871c700a2a20c15f863bd48dab_l3.png" alt="n = ((mu - Xmin)(Xmax - mu) - rho sigma^2) / (sigma^2 (1 - rho))">
</picture>
</div>

μ and σ² are the mean and variance of the observed scores, and ρ is the reliability. X<sub>min</sub> and X<sub>max</sub> are the lowest and highest ***possible*** scores on the test.

BB-CLASS instead uses the lowest and highest ***observed*** scores. If a test can be scored from 0 to 100 but the observed scores range from 10 to 90, BB-CLASS uses 10 and 90. This biases the effective test length and, through it, the rescaled scores, the true-score distribution, and every classification accuracy and consistency estimate. (The BB-CLASS manual's own LL example uses the observed range, 101 to 190.)

BBClassify asks for the possible minimum and maximum score instead.

## Running BBClassify

**Windows program.** Build `BBClassify.exe` as described under [Building the Windows program](#building-the-windows-program), or download it from the [releases page](https://github.com/HaakonEidemHaakstad/BBClassify_Python/releases) once a release is published. Then double-click it. As with BB-CLASS, it asks for the name of the control-card file. Put the control cards and data next to the program, or type the full path.

**Python.** With Python 3.10 or later:

```
pip install git+https://github.com/HaakonEidemHaakstad/BBClassify_Python
bbclassify            # prompts for the control-card file
bbclassify ccHB       # or give it directly
```

From Python code, `bbclassify.run("ccHB")` runs an analysis and returns the results.

The `examples` folder holds the two examples from the BB-CLASS manual.

## Control cards

A control-card file has three lines, exactly as for BB-CLASS (see the [BB-CLASS manual](bb-class_for_pc/BB-CLASS-man.pdf), pages 4–7). Fields are separated by spaces or tabs.

**Card 1: procedures**

```
type  r  nparm  [xfit a]  [Gquad | EHquad | EDquad]  [npts b]  [check]
```

| Field | Meaning |
|---|---|
| `type` | `HB` (Hanson and Brennan) or `LL` (Livingston and Lewis) |
| `r` | Reliability. For HB, 0 gives the binomial error model and r > 0 Lord's two-term compound-binomial model. For LL, 0 < r < 1. |
| `nparm` | 2 or 4: the two- or four-parameter beta true-score distribution |
| `xfit a` | Optional. The chi-square statistics use only cells with fitted frequency greater than a (default 0). |
| `EDquad`, `EHquad`, `Gquad` | Optional. The numerical integration method (default `EDquad`). |
| `npts b` | Optional. Number of quadrature intervals for EDquad and EHquad (default 1000). |
| `check` | Optional. Reports the sum of the bivariate probabilities, which should be 1. |

**Card 2: input data**

The file name goes in double quotation marks, followed by the data type: `F` for a frequency distribution, `R` for raw scores, or `M` for moments.

| Procedure | Frequencies | Raw scores | Moments |
|---|---|---|---|
| HB | `"data" F xcol fcol` | `"data" R xcol nitems` | `"data" M` |
| LL | `"data" F xcol fcol xmax [xmin]` | `"data" R xcol xmax [xmin]` | `"data" M` |

`xcol` and `fcol` are the columns holding the scores and frequencies, and `nitems` is the number of items.

**`xmax` and `xmin` are the only additions to BB-CLASS.** They are the highest and lowest possible scores on the test. `xmin` defaults to 0. For HB a frequency file must have one line for every score from 0 to the number of items, as in BB-CLASS.

A moments file contains these values, on as many lines as you like: the number of examinees, the mean, SD, skewness and kurtosis (not excess kurtosis), the minimum and maximum score, and the proportion of examinees in each category. For HB the minimum and maximum are 0 and the number of items. **For LL they must be the minimum and maximum *possible* scores.** BBClassify cannot check this, so it adds a reminder to the output.

**Card 3: cut scores**

```
K  xcut[1] ... xcut[K-1]  [tcut[1] ... tcut[K-1]]
```

These are the number of categories, the observed cut scores, and optionally the true cut scores on the proportion-correct scale. A score at or above a cut is placed in the category above it. Without true cut scores, `tcut = (xcut - Xmin) / (Xmax - Xmin)`.

**Examples**

```
HB 0. 4 check
"act288m" f 1 2
2   24.
```

```
LL 0.9 4 check
"LL data" f 1 2 190 101
3  140. 160.   .4  .6
```

The second example uses 101 and 190 as the possible range only to reproduce the BB-CLASS manual. For a real test you would give the test's possible range, for example `200 0`.

## Output

BBClassify writes three files next to the control cards, as BB-CLASS does:

* `<cc> out.txt` holds the main results, in BB-CLASS's layout:
  - the input, the effective test length and the category proportions;
  - the beta parameter estimates and number of moments fit;
  - the raw, fitted and true-score moments;
  - the chi-square statistics, the reliability and SEM;
  - the four accuracy and consistency tables;
  - any warnings and notes.
* `<cc> true dist.txt` holds the true-score distribution: midpoint, height, interval probability and cdf for each interval.
* `<cc> observed dist.txt` holds the observed and fitted score distributions.

Here `<cc>` is the name of the control-card file.

## Feedback on errors and warnings

* **Errors.** Every input error names the control card or data file and the line, says what is wrong, and says how to fix it. Examples are a missing possible maximum score, scores outside the possible range, cut scores outside the score range, non-numeric data, and a frequency file that skips scores.
* **Warnings.** Warnings flag results that may be unreliable. Examples are a negative Lord's k, negative two-term probabilities, a bivariate-probability sum that is not 1, a quadrature method that is inaccurate for the fitted distribution, empty categories, too few chi-square cells, and small samples.
* **Notes.** Notes record decisions made on your behalf, such as defaults or the fallback from a four- to a three-moment fit.

Warnings and notes are printed on screen and written to the output file. If BBClassify fails for any other reason, it saves the details to `BBClassify error log.txt` and asks you to report the problem.

## Differences from BB-CLASS

1. **LL score range.** The LL procedure uses the possible score range rather than the observed one, which is the correction this program exists for.
2. **Chi-square degrees of freedom.** They are the number of cells − 1 − the number of moments fit, as in Lord (1965). BB-CLASS reports one more degree of freedom. P-values are added.
3. **Splitting at true cuts.** Quadrature intervals that contain a true cut score are split at the cut. This reproduces the accuracy tables in the BB-CLASS manual exactly.
4. **Practical changes.**
   - Output files have a `.txt` extension.
   - The data file is also looked for next to the control cards.
   - Mac, Unix and Windows line endings are all accepted.
5. **Checks and messages.** BBClassify validates all input and reports problems as described above.

## Validation

The tests in `tests/` reproduce every number printed in the BB-CLASS manual's two examples. That covers the parameter estimates, the moments, the chi-squares, reliability and SEM, and all four accuracy and consistency tables for both the HB and the LL procedures. The tests also check that:

* raw, frequency and moments input give the same results;
* the possible score range is what enters the effective test length;
* invalid input gives an informative error.

Run them with `pip install -e .[test]` and `pytest`.

## Building the Windows program

```
pip install pyinstaller
pyinstaller BBClassify.spec
```

## License

MIT. See [LICENSE.md](LICENSE.md). BB-CLASS and its manual, in `bb-class_for_pc`, are included for reference and remain the property of their author.
