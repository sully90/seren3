def _get_predict_fname(out_dir, ioutput, weight, pdf_sampling):
    if pdf_sampling:
        return "%s/fb_%05i_%s_ftidal_pdf_sampling.predict" % (out_dir, ioutput, weight)
    return "%s/fb_%05i_%s.predict" % (out_dir, ioutput, weight)

def _get_results_fname(out_dir, ioutput, weight, NN, pdf_sampling):
    if pdf_sampling:
        return "%s/fb_%05i_NN%i_%s_ftidal_pdf_sampling.results" % (out_dir, ioutput, NN, weight)    
    return "%s/fb_%05i_NN%i_%s.results" % (out_dir, ioutput, NN, weight)

def main(ioutputs, weights_fname, NN):
    import os
    from seren3.utils import which

    _BIN = which("neural-net")
    _CWD = os.getcwd()
    _WEIGHT = "mw"

    def _run(weights_fname, predict_fname, results_fname):
        exe = "%s -l %s -p %s -o %s" % (_BIN, weights_fname, predict_fname, results_fname)
        # print exe
        os.system(exe)

    for ioutput in ioutputs:
        dir_name = "./%i_final/" % (ioutput)

        # Run the standard and tidal force PDF sampled models
        predict_fname = _get_predict_fname(dir_name, ioutput, _WEIGHT, False)
        predict_pdf_sampled_fname = _get_predict_fname(dir_name, ioutput, _WEIGHT, True)

        results_fname = _get_results_fname(dir_name, ioutput, _WEIGHT, NN, False)
        results_pdf_sampled_fname = _get_results_fname(dir_name, ioutput, _WEIGHT, NN, True)

        for pf, rf in zip([predict_fname, predict_pdf_sampled_fname], [results_fname, results_pdf_sampled_fname]):
            _run(weights_fname, pf, rf)


if __name__ == "__main__":
    import sys, os
    ioutputs = [42, 48, 60, 70, 80, 90, 100, 106]

    NN = int(sys.argv[1])
    weights_fname = "/lustre/scratch/astro/ds381/simulations/baryfrac/bc03_fesc2_nohm/neural-net2/NN%i.weights" % NN
    # weights_fname = "/lustre/scratch/astro/ds381/simulations/baryfrac/bc03_fesc2_nohm/neural-net2/106_90_60_NN40_niter100_fb_scaled_ftidal_scaled_logall.weights"
    
    # ioutputs = [int(i) for i in sys.argv[1:-2]]
    # weights_fname = sys.argv[-2]
    # NN = int(sys.argv[-1])

    # paths = ["bc03_fesc1.5_nohm/", "bc03_fesc2_nohm/", "bc03_fesc5_nohm/"]
    # paths = ["bc03_fesc2_nohm/", "bc03_fesc5_nohm/"]
    paths = ["bc03_fesc2_nohm/"]
    lustre_path = "/lustre/scratch/astro/ds381/simulations/baryfrac/"

    full_paths = ["%s/%s/neural-net2/" % (lustre_path, p) for p in paths]

    print "NN = ", NN
    print "Using weights file: ", weights_fname

    print ioutputs

    for fp in full_paths:
        os.chdir(fp)
        print "**************************************************"
        print os.getcwd()
        print "**************************************************"
        main(ioutputs, weights_fname, NN)

    print "Done"