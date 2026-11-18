
public class MLPRegressionAltura {

    private enum Activation { IDENTITY, LOGISTIC, RELU, TANH, SOFTMAX }

    private MLPRegressionAltura.Activation hidden;
    private MLPRegressionAltura.Activation output;
    private double[][] network;
    private double[][][] weights;
    private double[][] bias;
    public double predictTeste;

    public MLPRegressionAltura(String hidden, int[] layers, double[][][] weights, double[][] bias) {
        this.hidden = MLPRegressionAltura.Activation.valueOf(hidden.toUpperCase());

        this.network = new double[layers.length + 1][];
        for (int i = 0, l = layers.length; i < l; i++) {
            this.network[i + 1] = new double[layers[i]];
        }
        this.weights = weights;
        this.bias = bias;
    }

    public MLPRegressionAltura(String hidden,  int neurons, double[][][] weights, double[][] bias) {
        this(hidden, new int[] { neurons }, weights, bias);
    }

    private double[] compute(MLPRegressionAltura.Activation activation, double[] v, int nLayers) {
        switch (activation) {
            case LOGISTIC:
                if (nLayers > 1) {
                    for (int i = 0, l = v.length; i < l; i++) {
                        v[i] = 1. / (1. + Math.exp(-v[i]));
                    }
                } else {
                    for (int i = 0, l = v.length; i < l; i++) {
                        if (v[i] > 0) {
                            v[i] = -Math.log(1. + Math.exp(-v[i]));
                        } else {
                            v[i] = v[i] - Math.log(1. + Math.exp(-v[i]));
                        }
                    }
                }
                break;
            case RELU:
                for (int i = 0, l = v.length; i < l; i++) {
                    v[i] = Math.max(0, v[i]);
                }
                break;
            case TANH:
                for (int i = 0, l = v.length; i < l; i++) {
                    v[i] = Math.tanh(v[i]);
                }
                break;
            case SOFTMAX:
                double max = Double.NEGATIVE_INFINITY;
                for (double x : v) {
                    if (x > max) {
                        max = x;
                    }
                }
                for (int i = 0, l = v.length; i < l; i++) {
                    v[i] = Math.exp(v[i] - max);
                }
                double sum = 0.;
                for (double x : v) {
                    sum += x;
                }
                for (int i = 0, l = v.length; i < l; i++) {
                    v[i] /= sum;
                }
                break;
        }
        return v;
    }

    public double predict(double[] neurons) {
        this.network[0] = neurons;

        for (int i = 0; i < this.network.length - 1; i++) {
            for (int j = 0; j < this.network[i + 1].length; j++) {
                this.network[i + 1][j] = this.bias[i][j];
                for (int l = 0; l < this.network[i].length; l++) {
                    this.network[i + 1][j] += this.network[i][l] * this.weights[i][l][j];
                }
            }
            if ((i + 1) < (this.network.length - 1)) {
                this.network[i + 1] = this.compute(this.hidden, this.network[i + 1], this.network.length);
            }
        }

        //if (this.network[this.network.length - 1].length > 1) {
        // return this.network[this.network.length - 1];
        //}
        return this.network[this.network.length - 1][0];

    }




    public static double main(String[] args) {
        if (args.length == 2) {

            // Features:
            double[] features = new double[args.length];
            for (int i = 0, l = args.length; i < l; i++) {
                features[i] = Double.parseDouble(args[i]);
            }

            // Parameters:
            int[] layers = {3, 50, 1};
            double[][][] weights = {{{0.39406219, -0.7076449 ,  0.64819238},

                    {0.84055178,  1.12482126, -0.03750334}},

                    {{0.16313493, -0.31750795, -0.11758021, -0.32195575,  0.25874955,
                            -0.07632908,  0.13764449, -0.14969448, -0.14332069, -0.34913871,
                            -0.11715474,  0.32470303, -0.17025115, -0.11212933,  0.30419232,
                            -0.33205133, -0.26793143,  0.07012137,  0.22387136, -0.30188796,
                            0.18047255,  0.20052084, -0.28525709,  0.10588782, -0.02676595,
                            0.00748707,  0.28910621, -0.30638034, -0.17465794,  0.12402266,
                            -0.09582681, -0.13949986,  0.13128573,  0.29524358, -0.14248615,
                            -0.04731579, -0.15858308, -0.07379392,  0.08780417,  0.16486699,
                            -0.31254028,  0.09768844,  0.00757249, -0.2461587 , -0.27425693,
                            -0.1941593 , -0.236437  ,  0.27452869,  0.1430801 , -0.15832842},

                            {0.215607  , -0.1047658 , -0.16468711,  0.10376315,  0.06613563,
                                    -0.38303615, -0.140958  ,  0.11450093,  0.33230283,  0.22199513,
                                    0.1978103 ,  0.14146073, -0.15870894,  0.43437574,  0.03915471,
                                    -0.33866079,  0.40366591, -0.42710099, -0.32191906,  0.2109594 ,
                                    0.18655632,  0.20765509, -0.16846479, -0.40019738,  0.37800006,
                                    -0.08542286, -0.38250563, -0.03548565, -0.26519581,  0.24796748,
                                    -0.16111095,  0.47025963,  0.25781173, -0.16467761, -0.43038313,
                                    0.21420433, -0.11537375, -0.17293162, -0.04728769, -0.00861435,
                                    -0.10914643,  0.00118605,  0.23783174, -0.30551744, -0.3242412 ,
                                    0.17857432, -0.07550768,  0.01787395, -0.16265682, -0.33791449},

                            {-0.33146456,  0.21367475, -0.2363842 , -0.24288544, -0.20964705,
                                    -0.0867139 , -0.30789639, -0.19430212, -0.05112183,  0.19187212,
                                    -0.19235115,  0.05430738,  0.26185893, -0.0172534 , -0.09997883,
                                    -0.29621845,  0.18040751,  0.2784301 , -0.25140894, -0.23756233,
                                    0.19169449,  0.06713831, -0.05117668, -0.18181209, -0.42888604,
                                    0.02135739,  0.19722434,  0.32621246,  0.2162541 , -0.08616875,
                                    0.17124628, -0.50603036,  0.02892878,  0.12629207, -0.22053532,
                                    -0.06121547, -0.24849271, -0.17916906, -0.1999072 , -0.11333506,
                                    -0.22469085,  0.14336199, -0.30731692,  0.10942875, -0.16868131,
                                    -0.01042812, -0.19030053, -0.06053434,  0.00802694, -0.18461588}},

                    {{ 0.23829815},
                            {-0.3494703 },
                            { 0.14743424},
                            {-0.14164452},
                            { 0.34527817},
                            {-0.40791781},
                            { 0.01131551},
                            { 0.10534788},
                            { 0.14525772},
                            {-0.13955301},
                            { 0.05041123},
                            { 0.18474646},
                            { 0.17677095},
                            { 0.47441469},
                            { 0.16407657},
                            {-0.23075294},
                            { 0.33400814},
                            {-0.38449685},
                            {-0.23773565},
                            { 0.21594752},
                            { 0.16181415},
                            {-0.07451742},
                            {-0.27580444},
                            {-0.42870616},
                            { 0.14890782},
                            { 0.16782267},
                            {-0.27333803},
                            {-0.05160895},
                            {-0.37695517},
                            { 0.00659659},
                            { 0.01125003},
                            { 0.14752575},
                            { 0.36407599},
                            { 0.08738887},
                            {-0.21752715},
                            {-0.15481561},
                            { 0.04757129},
                            { 0.16769823},
                            { 0.1683363 },
                            {-0.23926724},
                            {-0.25131978},
                            {-0.20036666},
                            { 0.00761977},
                            {-0.22012512},
                            {-0.3434129 },
                            { 0.03935705},
                            { 0.22566353},
                            {-0.19146264},
                            { 0.00497192},
                            {-0.38120399}


                    }};
            double[][] bias = {{0.39591962, 0.42358013, 0.41439038}, {-0.23245127,  0.23344296, -0.07996315, -0.24307203,  0.18584864,
                    0.25470253,  0.22507578,  0.1407593 , -0.23375799,  0.31851571,
                    0.06066265,  0.26374773,  0.03334602, -0.30594514,  0.31954477,
                    -0.28256785,  0.07859256, -0.05293901, -0.17283387, -0.21780498,
                    0.11715389, -0.13850585,  0.06818238,  0.2767745 , -0.11684565,
                    -0.28692008, -0.04187006,  0.13445082,  0.28051741, -0.23274751,
                    -0.03956963, -0.04388512, -0.19099783, -0.14606073,  0.164829  ,
                    -0.28161431,  0.26041699,  0.23335768,  0.11631628, -0.13246037,
                    -0.24746037,  0.30028159, -0.29575501, -0.00472531, -0.10855464,
                    -0.22459267,  0.32623524, -0.19075451,  0.13913806,  0.22921152}, {-0.31226971}};
            // Prediction:
            MLPRegressionAltura clf = new MLPRegressionAltura("tanh", layers, weights, bias);
            double estimation = clf.predict(features);
            //System.out.println(estimation);
            return estimation;

        }
        return 0;
    }

}
