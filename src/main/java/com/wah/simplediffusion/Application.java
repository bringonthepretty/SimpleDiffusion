package com.wah.simplediffusion;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.MultiDataSet;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class Application {
    private static final String MODEL_ROOT_DIR = "H:\\Java\\Projects\\SimpleDiffusion\\models";

    private static final String DATASET_FOLDER_PATH = "H:\\Java\\Resources\\SimpleDiffusionResources\\dataset";

    static {
        System.loadLibrary("opencv_java470");
    }

    public static void main(String[] args) throws IOException {
        Diffusion diffusion = new Diffusion();

        List<MultiDataSet> multiDataSetList;

        long loadingStart = System.currentTimeMillis();
        System.out.println("model loading start");

        UNet unet = new UNet();
        ComputationGraph model = unet.getNeuralNetworkAsGraph();

        CheckpointListener saveListener = new CheckpointListener.Builder(MODEL_ROOT_DIR)
                .deleteExisting(true)
                .keepLast(2)
                .saveEveryNIterations(2000)
                .build();

        model.setListeners(new ScoreIterationListener(10), saveListener);

        long loadingEnd = System.currentTimeMillis();

        System.out.println("model loaded, took " + (loadingEnd - loadingStart) + " millis, training started");

        for (int i = 0; i < 10; i++) {
            System.out.println("preparing new dataset");
            multiDataSetList = diffusion.getMultiDataSet(DATASET_FOLDER_PATH, 800, 16, (int)(System.currentTimeMillis() % Integer.MAX_VALUE));
            System.out.println("new dataset prepared");

            for (int j = 0; j < 10; j++) {
                try {
                    multiDataSetList.forEach(model::fit);
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                    break;
                }
            }
        }

        model.save(new File("H:\\Java\\Projects\\SimpleDiffusion\\models\\model.zip"));
    }
}
