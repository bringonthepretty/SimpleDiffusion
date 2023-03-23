package com.wah.simplediffusion;

import org.apache.commons.collections4.ListUtils;
import org.apache.commons.lang3.tuple.Triple;
import org.apache.commons.math3.util.Pair;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.rng.CudaNativeRandom;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class Diffusion {

    private static final Double S = 0.008;

    private final Integer noiseSteps = 300;
    public static final Integer imageSize = 64;
    public static final Integer imageChannels = 3;

    private static final Integer DEFAULT_NOISE_SEED = 3213;

    private static final Double BETA_MAX_VALUE = 0.999;

    private static final NativeImageLoader loader = new NativeImageLoader();

    private final List<Double> alphasOverline = calculateAlphasOverlineCos(noiseSteps);
    private final List<Double> betas = calculateBetas(alphasOverline);


    static {
        System.loadLibrary("opencv_java470");
    }

    public List<MultiDataSet> getMultiDataSet(String path, int samplesNum, int batchSize, int seed) throws IOException {

        List<MultiDataSet> result = new ArrayList<>();

        Random random = new Random(seed);

        samplesNum = (samplesNum / batchSize) * batchSize;

        String[] fileNames = new File(path).list();

        if (fileNames == null) {
            return new ArrayList<>();
        }

        List<String> imageFileNamesList = Arrays.stream(fileNames).collect(Collectors.toCollection(LinkedList::new));
        Collections.shuffle(imageFileNamesList);

        List<INDArray> imagesAsNDArrayList = new ArrayList<>();

        while (imageFileNamesList.size() < samplesNum) {
            imageFileNamesList.addAll(imageFileNamesList);
            Collections.shuffle(imageFileNamesList);
        }

        for (int i = 0; i < samplesNum; i++) {
            Mat image = Imgcodecs.imread(path + "\\" + imageFileNamesList.remove(0));
            INDArray imageAsNDArray = loader.asMatrix(image);
            imageAsNDArray = imageAsNDArray.reshape(new int[]{imageChannels, imageSize, imageSize});
            imagesAsNDArrayList.add(imageAsNDArray);
        }

        List<Triple<INDArray, INDArray, Float>> dataSetEntryList = imagesAsNDArrayList.stream()
                .map(imageAsNDArray -> {
                    int step = random.nextInt(noiseSteps);
                    Pair<INDArray, Float> pair = getNoisedImageToNoiseMeanPair(imageAsNDArray, step, seed);
                    return Triple.of(pair.getFirst(), getPositionEmbeddingForTimeStep(step, imageSize, 10000), pair.getSecond());
                })
                .collect(Collectors.toCollection(ArrayList::new));

        List<List<Triple<INDArray, INDArray, Float>>> dataSetEntriesBatchesList = ListUtils.partition(dataSetEntryList, batchSize);

        dataSetEntriesBatchesList.forEach(dataSetEntryBatch -> {
            List<INDArray> featuresList = dataSetEntryBatch.stream().map(Triple::getLeft).toList();
            List<INDArray> positionEmbeddingsList = dataSetEntryBatch.stream().map(Triple::getMiddle).toList();
            List<Float> labelsList = dataSetEntryBatch.stream().map(Triple::getRight).toList();

            List<INDArray> labelsListAsINDArray = labelsList.stream().map(entry -> Nd4j.create(new float[]{entry})).toList();

            INDArray features = Nd4j.create(featuresList, featuresList.size(), imageChannels, imageSize, imageSize);
            INDArray positionEmbeddings = Nd4j.create(positionEmbeddingsList, positionEmbeddingsList.size(), imageChannels, imageSize, imageSize);
            INDArray labels = Nd4j.create(labelsListAsINDArray, labelsListAsINDArray.size(), 1);

            MultiDataSet multiDataSet = new MultiDataSet();
            multiDataSet.setFeatures(new INDArray[]{features, positionEmbeddings});
            multiDataSet.setLabels(new INDArray[]{labels});
            result.add(multiDataSet);

        });

        return result;
    }

    public Mat generateImage(ComputationGraph model, int seed) {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 65535);

        INDArray image = Nd4j.randn(seed, new long[]{1, imageChannels, imageSize, imageSize});
        for (int i = noiseSteps - 1; i > 1; i--) {
            INDArray[] predictedMean = model.output(image, getPositionEmbeddingForTimeStep(i, imageSize, 10000).reshape(1, imageChannels, imageSize, imageSize));
            System.out.println(predictedMean[0].getFloat(0));
            INDArray predictedNoise = Nd4j.randn(predictedMean[0].getFloat(0), betas.get(i), new long[]{imageChannels, imageSize, imageSize}, new CudaNativeRandom(DEFAULT_NOISE_SEED));

            INDArray noiseDuplicate = predictedNoise.dup();
            scaler.preProcess(noiseDuplicate);
            Imgcodecs.imwrite("C:\\Users\\bring\\Downloads\\results\\noise" + i + ".png", converter.convertToOrgOpenCvCoreMat(converter.convert(loader.asMat(noiseDuplicate))));

            image = removeNoiseFromImage(image, predictedNoise, i);
            image = getNoisedImage(image, predictedNoise, i - 1);
        }

        INDArray[] predictedMean = model.output(image, getPositionEmbeddingForTimeStep(0, imageSize, 10000).reshape(1, imageChannels, imageSize, imageSize));
        System.out.println(predictedMean[0].getFloat(0));
        INDArray predictedNoise = Nd4j.randn(predictedMean[0].getFloat(0), betas.get(0), new long[]{imageChannels, imageSize, imageSize}, new CudaNativeRandom(DEFAULT_NOISE_SEED));
        image = removeNoiseFromImage(image, predictedNoise, 0);

        scaler.preProcess(image);
        org.bytedeco.opencv.opencv_core.Mat byteDecoMat = loader.asMat(image);
        org.opencv.core.Mat mat = converter.convertToOrgOpenCvCoreMat(converter.convert(byteDecoMat));

        return mat;
    }

    /**
     * Returns a pair of image with random noise and noise itself.
     * @param image image to which noise will be applied
     * @param step current iteration, affects noise strength
     * @param seed noise seed
     * @return a pair of image with random noise and noise itself
     * 3DMatrix      3DMatrix                number              3DMatrix                  number
     * noisedImage =   image    *   sqrtAlphaCumulativeProduct  +  noise   *  sqrtOneMinusAlphaCumulativeProduct
     * noisedImage - noise * sqrtOneMinusAlphaCumulativeProduct = image * sqrtAlphaCumulativeProduct
     * (noisedImage - noise * sqrtOneMinusAlphaCumulativeProduct) / sqrtAlphaCumulativeProduct = image
     * image = (noisedImage - noise * sqrtOneMinusAlphaCumulativeProduct) / sqrtAlphaCumulativeProduct
     */
    private Pair<INDArray, INDArray> getNoisedImageToNoisePair(INDArray image, int step, int seed) {
        INDArray noise = Nd4j.randn(seed, image.shape());

        return new Pair<>(getNoisedImage(image, noise, step), noise);
    }

    /**
     * Returns a pair of image with noise and normal distribution mean.
     * @param image image to which noise will be applied
     * @param step current iteration, affects noise strength
     * @param seed normal distribution mean seed
     * @return a pair of image with noise and normal distribution mean.
     */
    private Pair<INDArray, Float> getNoisedImageToNoiseMeanPair(INDArray image, int step, int seed) {
        Random random = new Random(seed);
        CudaNativeRandom nativeRandom = new CudaNativeRandom(DEFAULT_NOISE_SEED);
        float mean = random.nextFloat() * 2 - 1; //expectation of the distribution

        INDArray noise = Nd4j.randn(mean, betas.get(step), image.shape(), nativeRandom);

        nativeRandom.close();

        return new Pair<>(getNoisedImage(image, noise, step), mean);
    }

    private INDArray getNoisedImage(INDArray image, INDArray noise, int step) {
        double sqrtAlphaCumulativeProduct = Math.sqrt(alphasOverline.get(step));
        double sqrtOneMinusAlphaCumulativeProduct = Math.sqrt(1 - alphasOverline.get(step));

        ImagePreProcessingScaler normalizationScaler = new ImagePreProcessingScaler(-1, 1);

        INDArray imageDuplicate = image.dup();
        INDArray noiseDuplicate = noise.dup();

        normalizationScaler.preProcess(imageDuplicate);

        INDArray a = imageDuplicate.muli(sqrtAlphaCumulativeProduct);
        INDArray b = noiseDuplicate.muli(sqrtOneMinusAlphaCumulativeProduct);
        INDArray result = a.add(b);

        return result;
    }

    /**
     * Removes noise from noised image.
     * @param noisedImage image from which noise will be removed
     * @param noise noise to be removed
     * @param step current iteration, affects noise strength
     * @return image without noise
     */
    private INDArray removeNoiseFromImage(INDArray noisedImage, INDArray noise, int step) {
        double sqrtAlphaCumulativeProduct = Math.sqrt(alphasOverline.get(step));
        double sqrtOneMinusAlphaCumulativeProduct = Math.sqrt(1 - alphasOverline.get(step));

        INDArray noisedArrayDuplicate = noisedImage.dup();
        INDArray noiseDuplicate = noise.dup();

        INDArray a = noiseDuplicate.muli(sqrtOneMinusAlphaCumulativeProduct);
        INDArray b = noisedArrayDuplicate.sub(a);
        INDArray result = b.divi(sqrtAlphaCumulativeProduct);

        return result;
    }

    private List<Double> calculateAlphasOverlineCos(int steps) {
        LinkedList<Double> result = new LinkedList<>();

        //f(0)
        double f0 = Math.pow(Math.cos((S / (S + 1)) * (Math.PI / 2)), 2);

        //f(0) / f(0)
        result.add(1d);

        for (int i = 1; i < steps; i++) {
            //f(t)
            double ft = Math.pow(Math.cos((((double)i / steps + S) / (S + 1)) * (Math.PI / 2)), 2);

            //f(t) / f(0)
            result.add(ft / f0);
        }

        return result;
    }

    private List<Double> calculateBetas(List<Double> alphasOverline) {
        List<Double> result = new LinkedList<>();

        //b = 1 - (a(t) / a(t-1))
        result.add(0d);
        for (int i = 1; i < noiseSteps; i++) {
            double beta = 1 - (alphasOverline.get(i) / alphasOverline.get(i - 1));
            result.add(Math.min(beta, BETA_MAX_VALUE));
        }

        return result;
    }

    /**
     * Returns position embedding as 3d array
     * @param positionEmbeddingLength length of result INDArray, must be even
     * @param n scalar, use 10000 if not sure
     * @return position embedding as 1d array
     */
    private INDArray getPositionEmbeddingForTimeStep(int position, int positionEmbeddingLength, int n) {
        if (positionEmbeddingLength % 2 == 1) {
            throw new IllegalArgumentException("position embedding length must be even");
        }

        float[][][] positionEmbeddingAsArray = new float[3][positionEmbeddingLength][positionEmbeddingLength];

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < positionEmbeddingLength; j++) {
                for (int k = 0; k < positionEmbeddingLength / 2; k++) {
                    double denominator = Math.pow(n, 2d * k / positionEmbeddingLength);
                    positionEmbeddingAsArray[i][j][k] = (float) Math.sin(position / denominator);
                    positionEmbeddingAsArray[i][j][k+1] = (float) Math.cos(position / denominator);
                }
            }
        }

        return Nd4j.create(positionEmbeddingAsArray);
    }
}
