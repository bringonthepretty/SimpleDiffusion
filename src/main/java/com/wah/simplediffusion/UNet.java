package com.wah.simplediffusion;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.deeplearning4j.nn.graph.ComputationGraph;
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

    private static final String INPUT_LAYER_NAME = "input";
    private static final String OUTPUT_LAYER_NAME = "output";
    private static final String INPUT_POSITION_EMBEDDING_LAYER_NAME = "input_position_embedding";
    private static final String INITIAL_CONVOLUTION_LAYER_NAME = "initial_convolution_layer";
    private static final String FINAL_CONVOLUTION_LAYER_NAME = "final_convolution_layer";

    private static final String CONVOLUTION_LAYER_BASE_NAME = "convolution_layer:";
    private static final String SUBSAMPLING_LAYER_BASE_NAME = "subsampling_layer:";
    private static final String UPSAMPLING_LAYER_BASE_NAME = "upsampling_layer:";
    private static final String BATCH_NORMALIZATION_LAYER_BASE_NAME = "batch_normalization_layer:";
    private static final String DROPOUT_LAYER_BASE_NAME = "dropout_layer:";

    private static final String SUBSAMPLING_BLOCK_NAME = "subsampling_block-";
    private static final String UPSAMPLING_BLOCK_NAME = "upsampling_block-";

    private final Deque<String> residualLayers = new ArrayDeque<>();

    public ComputationGraph getNeuralNetworkAsGraph() {

        ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .seed(COMPUTATION_GRAPH_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(1e-6))
                .weightInit(WeightInit.XAVIER)
                .miniBatch(true)
                .cacheMode(CacheMode.NONE)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .graphBuilder();

        Layer initialConvolutionLayer = new Convolution2D.Builder(3, 3)
                .stride(1, 1)
                .nIn(3)
                .nOut(downChannels[0])
                .padding(1, 1)
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                .activation(Activation.RELU)
                .build();

        Layer finalConvolutionLayer = new Convolution2D.Builder(1, 1)
                .nIn(upChannels[3])
                .nOut(3)
                .stride(1, 1)
                .convolutionMode(ConvolutionMode.Same)
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                .activation(Activation.RELU)
                .build();

        Layer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                .nIn(3)
                .nOut(OUTPUT_DIMENSION)
                .activation(Activation.IDENTITY)
                .build();

        graphBuilder.allowDisconnected(true);

        graphBuilder
                .addInputs(INPUT_LAYER_NAME, INPUT_POSITION_EMBEDDING_LAYER_NAME)
                .setInputTypes(InputType.convolutional(SIZE, SIZE, INPUT_DIMENSIONS), InputType.convolutional(SIZE, SIZE, INPUT_DIMENSIONS));

        graphBuilder.addLayer(INITIAL_CONVOLUTION_LAYER_NAME, initialConvolutionLayer, INPUT_LAYER_NAME, INPUT_POSITION_EMBEDDING_LAYER_NAME);

        addBlock(graphBuilder, downChannels[1], true);
        addBlock(graphBuilder, downChannels[2], true);
        addBlock(graphBuilder, downChannels[3], true);
        addBlock(graphBuilder, downChannels[4], true);

        addBlock(graphBuilder, upChannels[1], false);
        addBlock(graphBuilder, upChannels[2], false);
        addBlock(graphBuilder, upChannels[3], false);
        addBlock(graphBuilder, upChannels[4], false);

        graphBuilder.addLayer(FINAL_CONVOLUTION_LAYER_NAME, finalConvolutionLayer, graphBuilder.getLastAdded());
        graphBuilder.addLayer(OUTPUT_LAYER_NAME, outputLayer, graphBuilder.getLastAdded());

        graphBuilder.setOutputs(OUTPUT_LAYER_NAME);

        ComputationGraph model = new ComputationGraph(graphBuilder.build());
        model.init();

        return model;
    }

    /**
     * Adds block of layers to computation graph configuration builder
     * @param target target computation graph configuration builder
     * @param nOut number of layers output
     * @param blockType true for subsampling block, false for upsampling block
     */
    private void addBlock(ComputationGraphConfiguration.GraphBuilder target, int nOut, boolean blockType) {
        String baseName;
        int layerCounter = 1;

        if (blockType) {
            baseName = SUBSAMPLING_BLOCK_NAME;
            if (target.getLastAdded().contains(":") && !target.getLastAdded().contains(UPSAMPLING_BLOCK_NAME)) {
                layerCounter = Integer.parseInt(StringUtils.substringAfter(target.getLastAdded(), ":"));
                layerCounter++;
            }
        } else {
            baseName = UPSAMPLING_BLOCK_NAME;
            if (target.getLastAdded().contains(":") && !target.getLastAdded().contains(SUBSAMPLING_BLOCK_NAME)) {
                layerCounter = Integer.parseInt(StringUtils.substringAfter(target.getLastAdded(), ":"));
                layerCounter++;
            }
        }

        Layer significantLayer;
        if (blockType) {
            significantLayer = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(3, 3)
                    .stride(2, 2)
                    .padding(0, 0)
                    .dilation(1, 1)
                    .convolutionMode(ConvolutionMode.Same)
                    .build();
        } else {
            significantLayer = new Upsampling2D.Builder(2)
                    .build();
        }

        if (blockType) {
            target.addLayer(baseName + CONVOLUTION_LAYER_BASE_NAME + layerCounter++, getBasicConvolutionLayer(nOut, Activation.RELU), target.getLastAdded());
        } else {
            target.addLayer(baseName + CONVOLUTION_LAYER_BASE_NAME + layerCounter++, getBasicConvolutionLayer(nOut, Activation.RELU), target.getLastAdded(), residualLayers.removeLast());
        }

        target.addLayer(baseName + CONVOLUTION_LAYER_BASE_NAME + layerCounter++, getBasicConvolutionLayer(nOut, Activation.SELU), target.getLastAdded());
        target.addLayer(baseName + CONVOLUTION_LAYER_BASE_NAME + layerCounter++, getBasicConvolutionLayer(nOut, Activation.RELU), target.getLastAdded());

        if (blockType) {
            target.addLayer(baseName + SUBSAMPLING_LAYER_BASE_NAME + layerCounter++, significantLayer, target.getLastAdded());
            target.addLayer(baseName + DROPOUT_LAYER_BASE_NAME + layerCounter, new DropoutLayer.Builder(0.5).build(), target.getLastAdded());
            residualLayers.addLast(target.getLastAdded());
        } else {
            target.addLayer(baseName + UPSAMPLING_LAYER_BASE_NAME + layerCounter++, significantLayer, target.getLastAdded());
        }

        target.addLayer(baseName + BATCH_NORMALIZATION_LAYER_BASE_NAME + layerCounter++, getBasicBatchNormalizationLayer(nOut), target.getLastAdded());
        target.addLayer(baseName + BATCH_NORMALIZATION_LAYER_BASE_NAME + layerCounter, getBasicBatchNormalizationLayer(nOut), target.getLastAdded());
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
