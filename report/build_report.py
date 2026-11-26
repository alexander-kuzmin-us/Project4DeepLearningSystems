"""Writes report/module_summary.docx into the provided APA 7 template.

Tables are filled from outputs/metrics/*.csv so the report always matches the notebook run.
Run with a Python that has python-docx:  python report/build_report.py
"""
from pathlib import Path

import pandas as pd

from apa_docx import APADoc

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path.home() / "Downloads" / "apa-7-template-word-doc.docx"
OUT = ROOT / "report" / "module_summary.docx"
FIG = ROOT / "outputs" / "figures"
MET = ROOT / "outputs" / "metrics"

res = pd.read_csv(MET / "results_per_run.csv")
S = res.groupby("config").agg(["mean", "std"])


def ms(cfg, col, scale=1.0, d=2):
    return f"{S.loc[cfg, (col, 'mean')] * scale:.{d}f} ({S.loc[cfg, (col, 'std')] * scale:.{d}f})"


doc = APADoc(TEMPLATE)

doc.title_page(
    "Land-Use Classification of Sentinel-2 Satellite Images With a Convolutional Neural Network: "
    "A Controlled Test of Dropout",
    ["[Your Name]", "[Institution]", "[Course Name and Number]", "[Instructor]", "[Due Date]"],
)

doc.title("Land-Use Classification of Sentinel-2 Satellite Images With a Convolutional Neural Network: "
          "A Controlled Test of Dropout")

# ---------------------------------------------------------------------------
doc.heading1("Report Overview")
doc.body(
    "This report describes a deep learning experiment that classifies 64 × 64 pixel Sentinel-2 satellite "
    "images into 10 land-use classes, such as forest, highway, river, and permanent crops. I used the EuroSAT "
    "RGB dataset (Helber et al., 2019) and trained a convolutional neural network (CNN) in PyTorch. A controlled "
    "comparison tested whether adding dropout to the classifier head improves generalization, with every other "
    "setting held fixed and each configuration trained with three random seeds."
)

# ---------------------------------------------------------------------------
doc.heading1("Dataset and Task Description")
doc.body(
    "EuroSAT contains 27,000 labeled image patches cut from Sentinel-2 satellite imagery of 34 European "
    "countries. Each patch covers about 640 m × 640 m of ground and belongs to one of 10 land-use and land-cover "
    "classes (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, "
    "River, and SeaLake), with 2,000 to 3,000 images per class (Helber et al., 2019). The original product has 13 "
    "spectral bands. I used the RGB version, so every image is a 64 × 64 × 3 array of 8-bit pixel values. The "
    "data are real satellite observations, publicly available for research, and were downloaded through "
    "torchvision, as documented in the project's data/README.md file."
)
doc.body(
    "Before training I checked the data in the notebook. All images had the same shape, no labels were "
    "missing, and no two images were exact pixel duplicates. The classes are mildly imbalanced (largest to "
    "smallest class ratio of 1.5), so I used a stratified 70/15/15 split, giving 18,900 training, 4,050 "
    "validation, and 4,050 test images, and saved it to data/splits.csv for reproducibility. Pixels were "
    "standardized per color channel using the mean and standard deviation of the training split only, so no "
    "information from the validation or test images leaked into preprocessing. I deliberately used no data "
    "augmentation, because augmentation is itself a form of regularization (Goodfellow et al., 2016) and would have mixed with the effect of dropout in the experiment."
)
doc.body(
    "The dataset fits a single-label, multi-class image classification task: each patch has exactly one label, "
    "and classes are defined by spatial texture such as crop rows, road lines, building grids, and water "
    "surfaces. The inspection also showed a possible shortcut. Mean pixel brightness differs strongly by class "
    "(59.9 for Forest versus 127.7 for Industrial on a 0 to 255 scale), so a model could partly rely on color "
    "and brightness instead of structure."
)

# ---------------------------------------------------------------------------
doc.heading1("Model Architecture and Design Decisions")
doc.body(
    "I chose a CNN because convolutional layers apply the same small filters at every position of the image and "
    "pooling adds tolerance to small shifts, which matches imagery where a pattern such as a road or a field "
    "boundary can appear anywhere in the patch (Goodfellow et al., 2016). A recurrent network or a Transformer "
    "would have to learn this spatial structure from data, which is less efficient with 18,900 training images."
)
doc.body(
    "The baseline has three convolutional blocks. Each block applies two 3 × 3 convolutions, each followed by "
    "batch normalization and a ReLU activation, and ends with 2 × 2 max pooling. The number of channels doubles "
    "from 32 to 64 to 128 as the spatial size halves from 64 to 8 pixels. Two stacked 3 × 3 convolutions cover "
    "the same 5 × 5 region as one larger filter with fewer weights and an extra nonlinearity (Simonyan & Zisserman, 2015). After the last "
    "block, global average pooling turns each of the 128 feature maps into one number, so the classifier head "
    "receives 128 inputs instead of the 8,192 a flatten layer would produce. The head is a 256-unit fully "
    "connected layer with ReLU, a dropout layer whose rate is the experimental variable, and a final linear layer "
    "that outputs 10 class scores (logits); a softmax turns these into class probabilities. ReLU is used throughout "
    "because it is the recommended default activation for modern networks: it is cheap and keeps gradients large "
    "wherever a unit is active (Goodfellow et al., 2016). The model has 323,050 trainable parameters."
)
doc.body(
    "I chose three blocks as a balance between capacity and cost. Three rounds of pooling shrink the 64-pixel "
    "input to an 8 × 8 grid, so each final feature responds to a large part of the patch, which is needed to "
    "recognize field layouts and road networks. At 323,050 parameters the model is small relative to the 18,900 "
    "training images, and one 25-epoch run takes about five minutes on the Apple M4 GPU used here, which made the three "
    "seeds per configuration affordable. A deeper network was not needed for a baseline, since the goal was a "
    "correct, stable model rather than the highest possible accuracy."
)
doc.body(
    "Batch normalization standardizes each layer's inputs within a mini-batch, which allows higher learning "
    "rates and faster, more stable training (Ioffe & Szegedy, 2015). That mattered because the training budget "
    "was fixed at 25 epochs. The model was trained with cross-entropy loss, which is the negative log-likelihood "
    "of the correct class under the softmax output and the standard choice for single-label classification "
    "(Goodfellow et al., 2016), and the Adam optimizer with a learning rate of 0.001 and a batch size of 64. Adam adapts the "
    "step size for each parameter and works well with little tuning (Kingma & Ba, 2015), which suited the goal "
    "of a reasonable baseline rather than a tuned model. Every run trained for all 25 epochs so the full learning "
    "curves were visible, and the checkpoint with the lowest validation loss was kept for testing. This is the "
    "idea behind early stopping, which uses validation performance to pick the point before a model starts to "
    "overfit (Goodfellow et al., 2016), applied after training so the full curves could still be inspected. These choices "
    "worked in practice: training was stable, and the baseline's mean validation loss was still falling at "
    "epoch 25 (Figure 1)."
)

# ---------------------------------------------------------------------------
doc.heading1("Experimental Comparison")
doc.body(
    "The experiment changed exactly one setting: dropout with rate 0.5 before the final linear layer, compared "
    "with a rate of 0 (no dropout) in the baseline. During training, dropout randomly sets hidden units to zero, "
    "which prevents units from co-adapting and approximates averaging many thinned networks, and a rate of 0.5 "
    "works well for hidden layers across many tasks (Srivastava et al., 2014). Table 1 lists what changed and what "
    "stayed the same. The notebook enforces this in code: the experimental configuration is copied from the "
    "baseline configuration, and an assertion checks that only one key differs. Because dropout has no weights, "
    "the same seed gives both configurations identical starting weights and identical mini-batch order, so each "
    "pair of runs differs only in the dropout masks."
)
doc.table(
    1, "Controlled Comparison: What Changed and What Stayed the Same",
    ["Setting", "Baseline", "Experiment"],
    [["Dropout before final layer", "0.0 (none)", "0.5"],
     ["Architecture, parameters", "3 conv blocks, BN, GAP, 256-unit head", "Same"],
     ["Data split and normalization", "Stratified 70/15/15, train-only statistics", "Same"],
     ["Data augmentation", "None", "None"],
     ["Optimizer, learning rate, batch size", "Adam, 0.001, 64", "Same"],
     ["Epochs and checkpoint rule", "25, lowest validation loss", "Same"],
     ["Seeds (initial weights and batch order)", "0, 1, 2", "Same"]],
    [2.3, 2.4, 1.3],
    note="BN = batch normalization; GAP = global average pooling.",
)
doc.body(
    "Overfitting means a model fits its training data better than it generalizes to new data, which shows up as a "
    "gap between training and validation performance (Goodfellow et al., 2016). "
    "The comparison was meaningful for two reasons. First, the baseline curves showed a small but persistent "
    "gap between training and validation accuracy (97.4% versus 96.1% at the last epoch, averaged over seeds) and "
    "a noisy validation loss, which is the kind of behavior dropout is designed to reduce. Second, dropout acts on "
    "the only fully connected part of the network. Before the full training runs I wrote down the hypothesis that "
    "dropout would narrow the train-test gap and keep validation loss lower late in training, at the cost of "
    "slower convergence, and that the effect on accuracy might be small because batch normalization and global "
    "average pooling already regularize the model. I trained each configuration with three seeds because a "
    "difference below one percentage point could easily come from run-to-run noise. My plan called for checking the baseline curves first and switching to a learning-rate comparison if there was no overfitting. In practice both configurations were run together, and dropout was kept after the results were visible."
)

# ---------------------------------------------------------------------------
doc.heading1("Results and Interpretation")
doc.body(
    "Each model was evaluated once on the 4,050 held-out test images with two main metrics. Accuracy is easy to "
    "read but can hide a weak class. I therefore also report macro-F1: the F1 score combines precision and recall "
    "for a class into one number (Goodfellow et al., 2016), and macro-F1 averages it over the 10 classes without "
    "weighting, so a poorly recognized class lowers the score even though the classes differ in size. I also "
    "report the gap between training and test accuracy as a direct measure of overfitting, and per-class results "
    "and confusion matrices to see where errors occur."
)
gap = lambda cfg: ms(cfg, "gen_gap", 100)
doc.table(
    2, "Test Results by Configuration, Mean (SD) Over Three Seeds",
    ["Metric", "Baseline", "Dropout 0.5"],
    [["Test accuracy (%)", ms("baseline", "test_acc", 100), ms("dropout", "test_acc", 100)],
     ["Test macro-F1", ms("baseline", "test_macro_f1", 1, 3), ms("dropout", "test_macro_f1", 1, 3)],
     ["Train minus test accuracy (points)", gap("baseline"), gap("dropout")],
     ["Best epoch (lowest validation loss)", ms("baseline", "best_epoch", 1, 1), ms("dropout", "best_epoch", 1, 1)],
     ["Validation loss rise after its minimum", ms("baseline", "val_loss_rise", 1, 3), ms("dropout", "val_loss_rise", 1, 3)],
     ["Seconds per epoch", ms("baseline", "sec_per_epoch", 1, 1), ms("dropout", "sec_per_epoch", 1, 1)]],
    [2.8, 1.6, 1.6],
    note="Train accuracy is measured in evaluation mode on the same checkpoint used for testing. Test split: 4,050 images.",
)
doc.body(
    "Both configurations reached about 96% test accuracy (Table 2). The baseline averaged 96.37% and the dropout "
    "model 96.18%, and macro-F1 followed the same pattern (0.963 versus 0.961). Dropout scored higher on two of "
    "the three paired seeds, but the baseline produced the single best run (96.94%), and the mean paired "
    "difference of −0.19 percentage points is smaller than the spread between seeds. I therefore read the result "
    "as no measurable difference in test performance. The hypothesis was not supported."
)
doc.body(
    "The secondary measures are also inconclusive. The mean gap between training and test accuracy was smaller with dropout (1.44 versus 1.75 points), but the paired change was negative for seeds 0 and 1 and positive for seed 2. The standard deviation of test accuracy was lower (0.29 versus 0.50 points), but three seeds are too few to call this more stable. There was also no sign that dropout delayed overfitting: after reaching "
    "its minimum, validation loss rose slightly more with dropout (0.016 versus 0.005). Figure 1 shows why the "
    "effect was limited. The training and validation curves of the two models almost overlap, the dropout model's "
    "training loss is slightly higher as expected, and the baseline's mean validation loss was still at its lowest "
    "at the final epoch. In other words, the baseline was not overfitting much within 25 epochs, so there was "
    "little for dropout to fix. The architecture already had strong built-in regularization, and dropout only "
    "touched the 35,594 weights of the classifier head, about 11% of the model, while most capacity sits in the "
    "convolutional layers. Training time was essentially the same for both (Table 2)."
)
doc.figure(
    1, "Training and Validation Curves for Both Configurations (Mean and Standard Deviation Over Three Seeds)",
    FIG / "fig4_comparison_curves.png", 6.3,
    note="Dotted lines are training curves; solid (baseline) and dashed (dropout) lines are validation curves.",
)
doc.body(
    "The errors show where the real difficulty lies. Both models confused the same visually similar classes "
    "(Figure 2). For the baseline, pooled over three seeds, 48 of 1,350 HerbaceousVegetation predictions (3.6%) were PermanentCrop, 35 of 1,125 PermanentCrop predictions (3.1%) were AnnualCrop, and 21 of 1,125 River predictions (1.9%) were Highway. PermanentCrop had the lowest per-class F1 score (0.926 for the baseline and 0.934 with "
    "dropout), while SeaLake had the highest (0.990 and 0.986). At 10 m per pixel and in RGB only, orchards and vineyards, "
    "ordinary fields, and natural vegetation can share the same color and stripe texture, and a straight river "
    "or canal looks much like a road."
)
doc.figure(
    2, "Row-Normalized Test Confusion Matrices, Three Seeds Pooled",
    FIG / "fig6_confusion_matrices.png", 6.5,
    note="Rows are true classes and columns are predicted classes; diagonal values are per-class recall.",
)
doc.body(
    "A specific example of model behavior is how confident the models are when they are wrong. Figure 3 shows "
    "the baseline's 16 most confident test errors for seed 0. Three PermanentCrop patches with rectangular field "
    "patterns were labeled AnnualCrop with a confidence of 1.00, and several River patches were "
    "predicted as AnnualCrop, HerbaceousVegetation, Highway, or SeaLake with confidence of at least 0.97. One Industrial image "
    "(Industrial_1080.jpg), which shows a site with rows of round tanks, was predicted as Residential by the "
    "baseline with 0.98 confidence and by the dropout model with 1.00 confidence. Of the baseline's 162 errors, "
    "34 (21%) had a confidence above 0.9, compared with 30 of 147 errors for the dropout model. Modern neural networks are often overconfident (Guo et al., 2017), but on average this one was not: combining the confidence on correct (0.976) and wrong (0.720) predictions gives a mean of about 0.965, close to the 0.960 accuracy of that run. Wrong predictions are usually less confident, so the score is informative, but a threshold of 0.9 would still let these 34 errors through. Some of these errors are also arguably label "
    "ambiguity rather than model failure, because one label is assigned to a 640 m area that can contain several "
    "land uses."
)
doc.figure(
    3, "The Baseline Model's 16 Most Confident Test Errors (Seed 0)",
    FIG / "fig8_error_gallery.png", 5.2,
    note="Each panel shows the true class, the predicted class, and the model's confidence (highest softmax probability).",
)

# ---------------------------------------------------------------------------
doc.heading1("Limitations and Risks")
doc.body(
    "The dataset limits what the results can claim. All images come from Europe, so the 96% accuracy says "
    "nothing about other regions. Using only RGB discards 10 of the 13 Sentinel-2 bands, including near-infrared, "
    "which would likely help with the vegetation classes that caused most errors. Each 64 × 64 patch carries a "
    "single label even when it contains a mix of land uses, as several images in Figure 3 show."
)
doc.body(
    "The evaluation has its own limits. The split was random at the image level, not by location, so neighboring "
    "patches from the same area can appear in both training and test sets, and the test score likely describes "
    "performance on familiar landscapes rather than new ones. Three seeds and one data split are enough to show "
    "that the dropout effect is smaller than run-to-run noise, but not enough to estimate a small true effect "
    "precisely, and I did not run a formal significance test."
)
doc.body(
    "The training setup also shaped the conclusion. All runs used one fixed learning rate and a 25-epoch budget, "
    "and the validation loss was still improving at the end. The finding that dropout did not help therefore "
    "applies to this short, already regularized training regime. With longer training the baseline might start "
    "to overfit, and dropout could matter more."
)

# ---------------------------------------------------------------------------
doc.heading1("Ethical and Responsible Use")
doc.body(
    "The most specific risk is geographic bias. The model learned what crops, roads, and towns look like in "
    "Europe. If it were used to map land use in regions with different farming practices, such as small mixed "
    "plots, other crop types, or informal settlements, its error rate would likely rise, and those errors would "
    "fall on communities that are not represented in the training data. Anyone releasing such a model should "
    "document its intended region and use, and report performance broken down by the conditions it will face, "
    "as proposed in model cards (Mitchell et al., 2019)."
)
doc.body(
    "Confident errors turn this into a decision risk. If the model were used to check whether farmers' declared "
    "crop types match what is visible from space, a confident PermanentCrop-to-AnnualCrop error, the most common "
    "high-confidence mistake in Figure 3, could wrongly flag a farmer, and a 1.00 confidence score would make the "
    "error look certain. Calibration should be rechecked on any new region, where methods such as temperature scaling (Guo et al., 2017) may help, and high-stakes predictions should have human review regardless of confidence. Finally, land-use maps can be "
    "misused, for example to map settlements for surveillance. At 10 m resolution individuals cannot be "
    "identified, which limits privacy risk, but the Residential class could still help locate populated areas."
)

# ---------------------------------------------------------------------------
doc.heading1("Future Improvements")
doc.body(
    "Several next steps follow directly from the results. First, using all 13 spectral bands should help separate "
    "PermanentCrop, AnnualCrop, and HerbaceousVegetation, the classes behind most errors. Second, the next "
    "controlled experiment should target the training regime rather than overfitting, for example 50 or more "
    "epochs with a learning-rate schedule, or rotation and flip augmentation, since satellite images have no "
    "natural orientation. Third, transfer learning from a pretrained deep CNN is a realistic upgrade: Helber et "
    "al. (2019) report 98.57% accuracy on EuroSAT with deep CNNs, although their data split differs, so the numbers are not directly comparable."
)
doc.body(
    "On the evaluation side, a test set separated by location, plus a small test on non-European imagery, would "
    "measure how well the model transfers to new places. Running five to ten seeds would give confidence "
    "intervals for small effects like the one tested here. Reporting calibration error, especially on new regions, would show how far the confidence scores can be trusted for flagging uncertain predictions."
)

# ---------------------------------------------------------------------------
doc.page_break()
doc.heading1("References")
for ref in [
    "Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org",
    "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. "
    "*Proceedings of Machine Learning Research, 70*, 1321–1330. https://proceedings.mlr.press/v70/guo17a.html",
    "Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). EuroSAT: A novel dataset and deep learning benchmark "
    "for land use and land cover classification. *IEEE Journal of Selected Topics in Applied Earth Observations and "
    "Remote Sensing, 12*(7), 2217–2226. https://doi.org/10.1109/JSTARS.2019.2918242",
    "Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal "
    "covariate shift. *Proceedings of Machine Learning Research, 37*, 448–456. "
    "https://proceedings.mlr.press/v37/ioffe15.html",
    "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *3rd International Conference "
    "on Learning Representations*. https://doi.org/10.48550/arXiv.1412.6980",
    "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & "
    "Gebru, T. (2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency*. Association for Computing Machinery. "
    "https://doi.org/10.1145/3287560.3287596",
    "Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. In *3rd International Conference on Learning Representations*. https://doi.org/10.48550/arXiv.1409.1556",
    "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way "
    "to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. "
    "https://jmlr.org/papers/v15/srivastava14a.html",
]:
    doc.reference(ref)

doc.save(OUT)
print(f"wrote {OUT}")
