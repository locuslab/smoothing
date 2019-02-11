import pandas as pd

if __name__ == "__main__":
    output = pd.DataFrame(columns=["n", "correct, accurate", "correct, inaccuraet",
                                   "incorrect, accurate", "incorrect, inaccurate", "abstain"])

    results = []

    gold_standard = pd.read_csv("data/predict/imagenet/resnet50/noise_0.25/test/N_100000", delimiter="\t")[:450]
    for N in [100, 1000, 10000]:
        df = pd.read_csv("data/predict/imagenet/resnet50/noise_0.25/test/N_{}".format(N), delimiter="\t")[:450]
        accurate = df["predict"] == gold_standard["predict"]
        abstain = df["predict"] == -1
        frac_abstain = abstain.mean()
        frac_correct_accurate = (df["correct"] & accurate & ~abstain).mean()
        frac_correct_inaccurate =  (df["correct"] & ~accurate & ~abstain).mean()
        frac_incorrect_acccurate =  (~df["correct"] & accurate & ~abstain).mean()
        frac_incorrect_inacccurate =  (~df["correct"] & ~accurate & ~abstain).mean()
        results.append((N, frac_correct_accurate, frac_correct_inaccurate,
                        frac_incorrect_acccurate, frac_incorrect_inacccurate, frac_abstain))

    df = pd.DataFrame.from_records(results, "n", columns=["n", "correct, accurate", "correct, inaccurate",
                                                          "incorrect, accurate", "incorrect, inaccurate", "abstain"])
    print(df.to_latex(float_format=lambda f:"{:.2f}".format(f)))
