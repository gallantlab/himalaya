Model flowchart
---------------

The following flowchart can be used as a guide to select the right estimator.

.. mermaid::

  graph TD;
    A(How many feature space ?)
    O(Data size ?)
    M(Data size ?)
    OR(Hyperparameters ?)
    OK(Hyperparameters ?)
    MR(Hyperparameters ?)
    MK(Hyperparameters ?)

    A-- one-->O;
    A--multiple-->M;
    O--more samples-->OR;
    O--more features-->OK;
    M--more samples-->MR;
    M--more features-->MK;

    OK--known-->OKH[KernelRidge];
    OK--unknown-->OKCV[KernelRidgeCV];
    OR--known-->ORH[Ridge];
    OR--unknown-->ORCV[RidgeCV];
    MK--known-->MKH[WeightedKernelRidge];
    MK--unknown-->MKCV[MultipleKernelRidgeCV];

    MR--unknown-->MRCV[BandedRidgeCV];
    MR--known-->MKH;

    classDef fork fill:#FFDC97
    class A,O,M,OR,OK,MR,MK fork;

    classDef leaf fill:#ABBBE1
    class ORH,OKH,MRH,MKH leaf;
    class ORCV,OKCV,MRCV,MKCV leaf;

    click ORH "https://gallantlab.github.io/himalaya/_generated/himalaya.ridge.Ridge.html"
    click ORCV "https://gallantlab.github.io/himalaya/_generated/himalaya.ridge.RidgeCV.html"
    click MRCV "https://gallantlab.github.io/himalaya/_generated/himalaya.ridge.BandedRidgeCV.html"
    click OKH "https://gallantlab.github.io/himalaya/_generated/himalaya.kernel_ridge.KernelRidge.html"
    click OKCV "https://gallantlab.github.io/himalaya/_generated/himalaya.kernel_ridge.KernelRidgeCV.html"
    click MKH "https://gallantlab.github.io/himalaya/_generated/himalaya.kernel_ridge.WeightedKernelRidge.html"
    click MKCV "https://gallantlab.github.io/himalaya/_generated/himalaya.kernel_ridge.MultipleKernelRidgeCV.html"
