package com.wah.simplediffusion;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayDeque;
import java.util.Deque;

public class UNet {

    private static final Integer INPUT_DIMENSIONS = Diffusion.imageChannels;
    private static final Integer SIZE = Diffusion.imageSize;
    private static final Integer SAMPLING_SIZE_BASE = 64;
    private static final Integer[] downChannels = new Integer[]{SAMPLING_SIZE_BASE, SAMPLING_SIZE_BASE * 2, SAMPLING_SIZE_BASE * 4, SAMPLING_SIZE_BASE * 8, SAMPLING_SIZE_BASE * 16};
    private static final Integer[] upChannels = new Integer[]{SAMPLING_SIZE_BASE * 16, SAMPLING_SIZE_BASE * 8, SAMPLING_SIZE_BASE * 4, SAMPLING_SIZE_BASE * 2, SAMPLING_SIZE_BASE};
    private static final Integer OUTPUT_DIMENSION = 1;
    private static final Integer COMPUTATION_GRAPH_SEED = 1124124;
    private static final Double LEARNING_RATE = 1e-5;

    private static final String INPUT_LAYER_NAME = "input";
    private static final String OUTPUT_LAYER_NAME = "output";
    private static final String SOFTMAX_ACTIVATION_LAYER_NAME = "softmax_activation_layer";

    private static final String INPUT_POSITION_EMBEDDING_64_LAYER_NAME = "input_position_embedding_64";

    private static final String CONVOLUTION_LAYER_BASE_NAME = "convolution_layer:";
    private static final String SUBSAMPLING_LAYER_BASE_NAME = "subsampling_layer:";
    private static final String UPSAMPLING_LAYER_BASE_NAME = "upsampling_layer:";
    private static final String BATCH_NORMALIZATION_LAYER_BASE_NAME = "batch_normalization_layer:";
    private static final String DROPOUT_LAYER_BASE_NAME = "dropout_layer:";

    private static final String SUBSAMPLING_BLOCK_NAME = "subsampling_block-";
    private static final String UPSAMPLING_BLOCK_NAME = "upsampling_block-";
    private static final String MIDDLE_BLOCK_NAME = "middle_block-";

    private final Deque<String> residualLayers = new ArrayDeque<>();

    public ComputationGraph getNeuralNetworkAsGraph() {

        ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .seed(COMPUTATION_GRAPH_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(LEARNING_RATE))
                .weightInit(WeightInit.XAVIER)
                .miniBatch(true)
                .cacheMode(CacheMode.NONE)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .graphBuilder();

        Layer softmaxActivationLayer = new ActivationLayer(new ActivationSoftmax());

        Layer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(3)
                .nOut(OUTPUT_DIMENSION)
                .activation(Activation.IDENTITY)
                .build();

        graphBuilder.allowDisconnected(true);

        graphBuilder
                .addInputs(INPUT_LAYER_NAME, INPUT_POSITION_EMBEDDING_64_LAYER_NAME)
                .setInputTypes(InputType.convolutional(SIZE, SIZE, INPUT_DIMENSIONS), InputType.convolutional(SIZE, SIZE, INPUT_DIMENSIONS));

        addSubsamplingBlock(graphBuilder, downChannels[0]);
        addSubsamplingBlock(graphBuilder, downChannels[1]);
        addSubsamplingBlock(graphBuilder, downChannels[2]);
        addSubsamplingBlock(graphBuilder, downChannels[3]);

        addMiddleBlock(graphBuilder, downChannels[4]);

        addUpsamplingBlock(graphBuilder, upChannels[1]);
        addUpsamplingBlock(graphBuilder, upChannels[2]);
        addUpsamplingBlock(graphBuilder, upChannels[3]);
        addUpsamplingBlock(graphBuilder, upChannels[4]);

        graphBuilder.addLayer(SOFTMAX_ACTIVATION_LAYER_NAME, softmaxActivationLayer, graphBuilder.getLastAdded());
        graphBuilder.addLayer(OUTPUT_LAYER_NAME, outputLayer, graphBuilder.getLastAdded());

        graphBuilder.setOutputs(OUTPUT_LAYER_NAME);

        ComputationGraph model = new ComputationGraph(graphBuilder.build());
        model.init();

        System.out.println(model.summary());

        return model;
    }

    private void addSubsamplingBlock(ComputationGraphConfiguration.GraphBuilder target, int nOut) {
        String baseName = SUBSAMPLING_BLOCK_NAME;
        int layerCounter = 1;

        if (target.getLastAdded().contains(":")) {
            layerCounter = Integer.parseInt(StringUtils.substringAfter(target.getLastAdded(), ":"));
            layerCounter++;
        }

        Layer significantLayer = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .padding(0, 0)
                .dilation(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .build();

        target.addLayer(baseName + CONVOLUTION_LAYER_BASE_NAME + layerCounter++, getBasicConvolutionLayer(nOut, Activation.RELU), target.getLastAdded());
        target.addLayer(baseName + CONVOLUTION_LAYER_BASE_NAME + layerCounter++, getBasicConvolutionLayer(nOut, Activation.RELU), target.getLastAdded());
        target.addLayer(baseName + SUBSAMPLING_LAYER_BASE_NAME + layerCounter++, significantLayer, target.getLastAdded());
        target.addLayer(baseName + DROPOUT_LAYER_BASE_NAME + layerCounter++, new DropoutLayer.Builder(0.5).build(), target.getLastAdded());
        residualLayers.addLast(target.getLastAdded());
        target.addLayer(baseName + BATCH_NORMALIZATION_LAYER_BASE_NAME + layerCounter++, getBasicBatchNormalizationLayer(nOut), target.getLastAdded());
        target.addLayer(baseName + BATCH_NORMALIZATION_LAYER_BASE_NAME + layerCounter, getBasicBatchNormalizationLayer(nOut), target.getLastAdded());
    }

    private void addUpsamplingBlock(ComputationGraphConfiguration.GraphBuilder target, int nOut) {
        String baseName = UPSAMPLING_BLOCK_NAME;
        int layerCounter = 1;

        if (target.getLastAdded().contains(":")) {
            layerCounter = Integer.parseInt(StringUtils.substringAfter(target.getLastAdded(), ":"));
            layerCounter++;
        }

        Layer significantLayer = new Upsampling2D.Builder(2)
                .build();

        target.addLayer(baseName + CONVOLUTION_LAYER_BASE_NAME + layerCounter++, getBasicConvolutionLayer(nOut, Activation.RELU), target.getLastAdded());
        target.addLayer(baseName + CONVOLUTION_LAYER_BASE_NAME + layerCounter++, getBasicConvolutionLayer(nOut, Activation.RELU), target.getLastAdded(), residualLayers.removeLast());
        target.addLayer(baseName + UPSAMPLING_LAYER_BASE_NAME + layerCounter++, significantLayer, target.getLastAdded());
        target.addLayer(baseName + BATCH_NORMALIZATION_LAYER_BASE_NAME + layerCounter++, getBasicBatchNormalizationLayer(nOut), target.getLastAdded());
        target.addLayer(baseName + BATCH_NORMALIZATION_LAYER_BASE_NAME + layerCounter, getBasicBatchNormalizationLayer(nOut), target.getLastAdded());
    }

    private void addMiddleBlock(ComputationGraphConfiguration.GraphBuilder target, int nOut) {
        String baseName = MIDDLE_BLOCK_NAME;
        int layerCounter = 1;

        if (target.getLastAdded().contains(":")) {
            layerCounter = Integer.parseInt(StringUtils.substringAfter(target.getLastAdded(), ":"));
            layerCounter++;
        }

        target.addLayer(baseName + CONVOLUTION_LAYER_BASE_NAME + layerCounter++, getBasicConvolutionLayer(nOut, Activation.RELU), target.getLastAdded());
        target.addLayer(baseName + CONVOLUTION_LAYER_BASE_NAME + layerCounter, getBasicConvolutionLayer(nOut, Activation.RELU), target.getLastAdded());
    }

    /**
     * Returns new convolution layer with kernel size {3, 3}, stride {1, 1}, provided number of outputs, padding{1, 1},
     * CUDNN algorithm mode PREFER_FASTEST and provided activation
     * @param nOut number of outputs
     * @return new convolution layer
     */
    private Layer getBasicConvolutionLayer(int nOut, Activation activationType) {
        return new Convolution2D.Builder(3, 3)
                .stride(1, 1)
                .nOut(nOut)
                .padding(1, 1)
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                .activation(activationType)
                .build();
    }

    private Layer getBasicBatchNormalizationLayer(int nOut) {
        return new BatchNormalization.Builder()
                .nOut(nOut)
                .eps(1e-05)
                .decay(0.1)
                .build();
    }
}
