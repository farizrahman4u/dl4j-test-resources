Code to generate INDArray files:
```
    public static void main(String[] args) throws Exception {

        File f = new ClassPathResource("deeplearning4j-zoo/goldenretriever.jpg").getFile();
        String baseDir = "C:/DL4J/Git/dl4j-test-resources/src/main/resources/deeplearning4j-zoo";

        for(int size : new int[]{128, 224, 299}) {

            NativeImageLoader loader = new NativeImageLoader(size, size, 3);

            INDArray bgr224 = loader.asMatrix(f);

            File outFileBgr = new File(baseDir, "goldenretriever_bgr" + size + "_unnormalized_nchw_INDArray.bin");
            Nd4j.saveBinary(bgr224, outFileBgr);

            ImageTransform it = new ColorConversionTransform(COLOR_BGR2RGB);
            ImageWritable iw = loader.asWritable(f);
            iw = it.transform(iw);

            INDArray rgb = loader.asMatrix(iw);
            File outFileRgb224 = new File(baseDir, "goldenretriever_rgb" + size + "_unnormalized_nchw_INDArray.bin");
            Nd4j.saveBinary(rgb, outFileRgb224);
        }
    }
```