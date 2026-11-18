import java.awt.image.BufferedImage;
import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Locale;
import javax.imageio.ImageIO;

/**
 * Referencia em Java puro (sem Android) da parte de calculo do app Pastagem.
 *
 * A logica abaixo foi copiada de AlturaActivity.textura() e
 * BiomassaActivity.previsaoBiomassa(). As unicas trocas sao as de API:
 *   Bitmap.getPixel(j,i) / Color.red|green|blue  ->  BufferedImage.getRGB(j,i) + shifts
 * As classes MLPClassifier e MLPRegressionAltura sao as originais do app
 * (so foi removida a linha "package").
 *
 * Uso:  java PastagemRef <pasta_de_imagens> <saida.csv>
 */
public class PastagemRef {

    public static void main(String[] args) throws Exception {
        File dir = new File(args[0]);
        File[] files = dir.listFiles((d, n) -> {
            String s = n.toLowerCase();
            return s.endsWith(".jpg") || s.endsWith(".jpeg") || s.endsWith(".png");
        });
        Arrays.sort(files);
        try (PrintWriter out = new PrintWriter(args[1], "UTF-8")) {
            out.println("arquivo;largura;altura_px;entropia;homogeneidade;segundo_momento_angular;"
                    + "desvio_padrao_cinza;area_escura_pct;gli_medio;"
                    + "altura_mlp_cm;altura_linear_cm;biomassa_kg_ha");
            for (File f : files) {
                BufferedImage bitmap = ImageIO.read(f);
                double[] r = processa(bitmap);
                StringBuilder sb = new StringBuilder(f.getName());
                sb.append(';').append(bitmap.getWidth()).append(';').append(bitmap.getHeight());
                for (double v : r) sb.append(';').append(String.format(Locale.ROOT, "%.17g", v));
                out.println(sb);
            }
        }
    }

    static double[] processa(BufferedImage bitmap) {
        double avg2, r, g, b;
        double avg, GLIMedio, somaGLI = 0;
        int auxi = 0;
        double areaEscura, cont = 0, cont2 = 0, avgmedio = 0;
        double[][] coocorrencia = new double[8][8];

        //Matriz cocorrencia
        for (int i = 0; i < bitmap.getHeight(); i++) {
            for (int j = 0; j < bitmap.getWidth(); j++) {
                int pixel = bitmap.getRGB(j, i);
                r = (pixel >> 16) & 0xff;
                g = (pixel >> 8) & 0xff;
                b = pixel & 0xff;
                avg = (r + g + b) / 3;
                avgmedio = avgmedio + avg;
                cont2 = cont2 + 1;
                //Calcula GLI
                if ((2 * g + r + b) == 0) {
                    auxi = auxi + 1;
                } else {
                    somaGLI = (2 * g - r - b) / (2 * g + r + b) + somaGLI;
                }
                if (j < (bitmap.getWidth() - 1)) {
                    pixel = bitmap.getRGB(j + 1, i);
                    r = (pixel >> 16) & 0xff;
                    g = (pixel >> 8) & 0xff;
                    b = pixel & 0xff;
                    avg2 = (r + g + b) / 96;
                    coocorrencia[(int) avg / 32][(int) avg2] = coocorrencia[(int) avg / 32][(int) avg2] + 1;
                }
            }
        }

        GLIMedio = somaGLI / (cont2);
        avgmedio = avgmedio / cont2;
        double somaDP = 0;

        for (int i = 0; i < bitmap.getHeight(); i++) {
            for (int j = 0; j < bitmap.getWidth(); j++) {
                int pixel = bitmap.getRGB(j, i);
                r = (pixel >> 16) & 0xff;
                g = (pixel >> 8) & 0xff;
                b = pixel & 0xff;
                avg = (r + g + b) / 3;
                if ((avg) <= 30) {
                    cont = cont + 1;
                }
                somaDP = (float) (somaDP + Math.pow(avg - avgmedio, 2));
            }
        }
        somaDP = somaDP / (cont2);
        somaDP = (float) Math.sqrt(somaDP);

        double segundoMomentoAngular = 0;
        double entropia = 0;
        double homogeniedade = 0;
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                coocorrencia[i][j] = (coocorrencia[i][j]) / ((bitmap.getWidth() - 1) * bitmap.getHeight());
                segundoMomentoAngular = segundoMomentoAngular + Math.pow((coocorrencia[i][j]), 2);
                if ((coocorrencia[i][j]) != 0) {
                    entropia = entropia + (coocorrencia[i][j]) * (Math.log((coocorrencia[i][j])) / Math.log(2));
                }
                homogeniedade = homogeniedade + (coocorrencia[i][j]) / (1 + Math.pow(i - j, 2));
            }
        }
        areaEscura = 100 * ((double) cont / ((double) cont2));

        // AlturaActivity
        String[] dados = new String[]{String.valueOf(entropia), String.valueOf(homogeniedade)};
        double alturaMlp = MLPRegressionAltura.main(dados);
        alturaMlp = alturaMlp * 8.648518659284955 + 21.125;

        // BiomassaActivity
        double altura = -437.2714 + entropia * (-41.8438) + homogeniedade * (357.7855) + somaDP * (-0.24237) + areaEscura * (0.1394);
        dados = new String[]{String.valueOf(altura), String.valueOf(GLIMedio)};
        double Biomassa = MLPClassifier.main(dados) * 3916.226388256552 + 6428.8749844808635;

        return new double[]{entropia, homogeniedade, segundoMomentoAngular, somaDP, areaEscura, GLIMedio,
                alturaMlp, altura, Biomassa};
    }
}
