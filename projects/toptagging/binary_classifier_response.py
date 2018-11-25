from __future__ import division
from collections import OrderedDict
import numpy as np
import ROOT
from ROOT import TH1F
import os

# FIXME SetYTitle

class BinaryClassifierResponse(object):
    def __init__(self,
                 name,
                 title,
                 directory):

        self._name = name
        self._title = title
        self._directory = directory

        self._keys = [
            "test_signal",
            "train_signal",
            "test_background",
            "train_background",
        ]

        self._response = OrderedDict([(key, np.zeros((0, ))) for key in self._keys])
        self._path = os.path.join(directory, name + ".{ext}")

        self._train_sig_color = ROOT.kBlue
        self._train_bkg_color = ROOT.kRed
        self._test_sig_color = 38 # blue
        self._test_bkg_color = 46 # red
        self._train_marker = 21 # full square

    def append(self, y_true, y_score, is_train):
        if len(y_true) != len(y_score):
            raise ValueError

        is_sig = y_true.astype(np.bool)
        is_bkg = np.logical_not(is_sig)
 
        sig_response = y_score[is_sig]
        bkg_response = y_score[is_bkg]

        dset = "train" if is_train else "test"
        sig_key = dset + "_signal"
        bkg_key = dset + "_background"

        self._response[sig_key] = np.append(self._response[sig_key], sig_response)
        self._response[bkg_key] = np.append(self._response[bkg_key], bkg_response)

        
    def _draw(self):
        canvas = ROOT.TCanvas("c", "c", 1200, 800)
        canvas.cd()

        h0 = TH1F("untitled", self._title, 20, 0, 1)
        h0.SetXTitle("Model response")

        hists = OrderedDict([(key, TH1F(key, key, 20, 0, 1)) for key in self._keys])
        for key, hist in hists.iteritems():
            for each in self._response[key]:
                hist.Fill(each) 

        # Normalization
        for each in hists.values():
            each.Scale(1.0 / each.Integral())

        max_value = max(each.GetMaximum() for each in hists.values())
        h0.SetMaximum(1.4 * max_value)

        # Color
        for key, hist in hists.iteritems():
            is_train = "train" in key
            is_signal = "signal" in key

            if is_train:
                color = self._train_sig_color if is_signal else self._train_bkg_color
            else:
                color = self._test_sig_color if is_signal else self._test_bkg_color
            hist.SetLineColor(color)

            if is_train:
                hist.SetMarkerStyle(self._train_marker)
                hist.SetMarkerColor(color)

        hists["test_signal"].SetFillColorAlpha(self._test_sig_color, 0.333)
        hists["test_background"].SetFillColor(self._test_bkg_color)
        # FIXME attribute
        hists["test_background"].SetFillStyle(3354)

        # Draw
        h0.Draw("hist")
        for name, hist in hists.iteritems():
            if "train" in name:
                hist.Draw("E1 same")
            else:
                hist.Draw("hist same")

        # Legend
        legend = ROOT.TLegend(0.1, 0.7, 0.9, 0.9)
        legend.SetNColumns(2)
        for key, hist in hists.iteritems():
            dset, cls = key.split("_")
            label = "{} ({} sample)".format(cls.title(), dset)
            option = "pl" if "train" in key else "lf"
            legend.AddEntry(hist, label, option)
        legend.Draw()


        ROOT.gStyle.SetOptStat(False)

        canvas.SaveAs(self._path.format(ext="png"))
        canvas.SaveAs(self._path.format(ext="pdf"))



    def finish(self):
        np.savez(self._path.format(ext="npz"), **self._response)
        self._draw()


def _test():
    model_response = BinaryClassifierResponse(directory='./logs', name="test", title="TEST")

    for _ in range(100):
        y_true = np.random.randint(low=0, high=2, size=100)
        y_score = np.random.random(100)
        model_response.append(is_train=True, y_true=y_true, y_score=y_score)

    for _ in range(100):
        y_true = np.random.randint(low=0, high=2, size=100)
        y_score = np.random.random(100)
        model_response.append(is_train=False, y_true=y_true, y_score=y_score)

    model_response.finish()


if __name__ == "__main__":
    from ROOT import gROOT
    gROOT.SetBatch(True)
    _test()
