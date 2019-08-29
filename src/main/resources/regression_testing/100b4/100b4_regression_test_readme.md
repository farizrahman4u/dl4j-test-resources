

Files for regression testing.

Created with: DL4J 1.0.0-beta4 Release

This file provides the configurations and code used to generate these files.
These tests (except for the synthetic CNN and RNN) were pulled from the examples

/////////////////////////////////////////////////

# GravesLSTMCharModelingExample

```java
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Random;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class LSTMCharModellingExample {
    public static void main( String[] args ) throws Exception {
        int lstmLayerSize = 200;					//Number of units in each LSTM layer
        int miniBatchSize = 32;						//Size of mini batch to use when  training
        int exampleLength = 1000;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 1;							//Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
        int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
        int nCharactersToSample = 300;				//Length of each sample to generate
        String generationInitialization = null;		//Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
        Random rng = new Random(12345);

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our LSTM network.
        CharacterIterator iter = getShakespeareIterator(miniBatchSize,exampleLength);
        int nOut = iter.totalOutcomes();

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.005))
                .list()
                .layer(new LSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                        .nIn(lstmLayerSize).nOut(nOut).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.save(new File("output/GravesLSTMCharModelingExample_100b4.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{3, iter.inputColumns(), 10});
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/GravesLSTMCharModelingExample_Input_100b4.bin")))) {
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input);
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/GravesLSTMCharModelingExample_Output_100b4.bin")))) {
            Nd4j.write(output, dos);
        }
    }

    /** Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
     * DataSetIterator that does vectorization based on the text.
     * @param miniBatchSize Number of text segments in each training mini-batch
     * @param sequenceLength Number of characters in each text segment.
     */
    public static CharacterIterator getShakespeareIterator(int miniBatchSize, int sequenceLength) throws Exception {
        //The Complete Works of William Shakespeare
        //5.3MB file in UTF-8 Encoding, ~5.4 million characters
        //https://www.gutenberg.org/ebooks/100
        String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileLocation = tempDir + "/Shakespeare.txt";	//Storage location from downloaded file
        File f = new File(fileLocation);
        if( !f.exists() ){
            FileUtils.copyURLToFile(new URL(url), f);
            System.out.println("File downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing text file at " + f.getAbsolutePath());
        }

        if(!f.exists()) throw new IOException("File does not exist: " + fileLocation);	//Download problem?

        char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
        return new CharacterIterator(fileLocation, StandardCharsets.UTF_8,
                miniBatchSize, sequenceLength, validCharacters, new Random(12345));
    }
}
```

```java
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/** A simple DataSetIterator for use in the LSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next character in the sequence.<br>
 * This is done by randomly choosing a position in the text file, at offsets of 0, exampleLength, 2*exampleLength, etc
 * to start each sequence. Then we convert each character to an index, i.e., a one-hot vector.
 * Then the character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
 *
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */
public class CharacterIterator implements DataSetIterator {
    //Valid characters
    private char[] validCharacters;
    //Maps each character to an index ind the input/output
    private Map<Character, Integer> charToIdxMap;
    //All characters of the input file (after filtering to only those that are valid
    private char[] fileCharacters;
    //Length of each example/minibatch (number of characters)
    private int exampleLength;
    //Size of each minibatch (number of examples)
    private int miniBatchSize;
    private Random rng;
    //Offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @param rng Random number generator, for repeatability if required
     * @throws IOException If text file cannot  be loaded
     */
    public CharacterIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                             char[] validCharacters, Random rng) throws IOException {
        this(textFilePath,textFileEncoding,miniBatchSize,exampleLength,validCharacters,rng,null);
    }
    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @param rng Random number generator, for repeatability if required
     * @param commentChars if non-null, lines starting with this string are skipped.
     * @throws IOException If text file cannot  be loaded
     */
    public CharacterIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                             char[] validCharacters, Random rng, String commentChars) throws IOException {
        if( !new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
        if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
        this.validCharacters = validCharacters;
        this.exampleLength = exampleLength;
        this.miniBatchSize = miniBatchSize;
        this.rng = rng;

        //Store valid characters is a map for later use in vectorization
        charToIdxMap = new HashMap<>();
        for( int i=0; i<validCharacters.length; i++ ) charToIdxMap.put(validCharacters[i], i);

        //Load file and convert contents to a char[]
        boolean newLineValid = charToIdxMap.containsKey('\n');
        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(),textFileEncoding);
        if (commentChars != null) {
            List<String> withoutComments = new ArrayList<>();
            for(String line:lines) {
                if (!line.startsWith(commentChars)) {
                    withoutComments.add(line);
                }
            }
            lines=withoutComments;
        }
        int maxSize = lines.size();	//add lines.size() to account for newline characters at end of each line
        for( String s : lines ) maxSize += s.length();
        char[] characters = new char[maxSize];
        int currIdx = 0;
        for( String s : lines ){
            char[] thisLine = s.toCharArray();
            for (char aThisLine : thisLine) {
                if (!charToIdxMap.containsKey(aThisLine)) continue;
                characters[currIdx++] = aThisLine;
            }
            if(newLineValid) characters[currIdx++] = '\n';
        }

        if( currIdx == characters.length ){
            fileCharacters = characters;
        } else {
            fileCharacters = Arrays.copyOfRange(characters, 0, currIdx);
        }
        if( exampleLength >= fileCharacters.length ) throw new IllegalArgumentException("exampleLength="+exampleLength
                +" cannot exceed number of valid characters in file ("+fileCharacters.length+")");

        int nRemoved = maxSize - fileCharacters.length;
        System.out.println("Loaded and converted file: " + fileCharacters.length + " valid characters of "
                + maxSize + " total characters (" + nRemoved + " removed)");

        initializeOffsets();
    }

    /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
    public static char[] getMinimalCharacterSet(){
        List<Character> validChars = new LinkedList<>();
        for(char c='a'; c<='z'; c++) validChars.add(c);
        for(char c='A'; c<='Z'; c++) validChars.add(c);
        for(char c='0'; c<='9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for( char c : temp ) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i=0;
        for( Character c : validChars ) out[i++] = c;
        return out;
    }

    /** As per getMinimalCharacterSet(), but with a few extra characters */
    public static char[] getDefaultCharacterSet(){
        List<Character> validChars = new LinkedList<>();
        for(char c : getMinimalCharacterSet() ) validChars.add(c);
        char[] additionalChars = {'@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
                '\\', '|', '<', '>'};
        for( char c : additionalChars ) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i=0;
        for( Character c : validChars ) out[i++] = c;
        return out;
    }

    public char convertIndexToCharacter( int idx ){
        return validCharacters[idx];
    }

    public int convertCharacterToIndex( char c ){
        return charToIdxMap.get(c);
    }

    public char getRandomCharacter(){
        return validCharacters[(int) (rng.nextDouble()*validCharacters.length)];
    }

    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    public DataSet next() {
        return next(miniBatchSize);
    }

    public DataSet next(int num) {
        if( exampleStartOffsets.size() == 0 ) throw new NoSuchElementException();

        int currMinibatchSize = Math.min(num, exampleStartOffsets.size());
        //Allocate space:
        //Note the order here:
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
        INDArray input = Nd4j.create(new int[]{currMinibatchSize,validCharacters.length,exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize,validCharacters.length,exampleLength}, 'f');

        for( int i=0; i<currMinibatchSize; i++ ){
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            int currCharIdx = charToIdxMap.get(fileCharacters[startIdx]);	//Current input
            int c=0;
            for( int j=startIdx+1; j<endIdx; j++, c++ ){
                int nextCharIdx = charToIdxMap.get(fileCharacters[j]);		//Next character to predict
                input.putScalar(new int[]{i,currCharIdx,c}, 1.0);
                labels.putScalar(new int[]{i,nextCharIdx,c}, 1.0);
                currCharIdx = nextCharIdx;
            }
        }

        return new DataSet(input,labels);
    }

    public int totalExamples() {
        return (fileCharacters.length-1) / miniBatchSize - 2;
    }

    public int inputColumns() {
        return validCharacters.length;
    }

    public int totalOutcomes() {
        return validCharacters.length;
    }

    public void reset() {
        exampleStartOffsets.clear();
        initializeOffsets();
    }

    private void initializeOffsets() {
        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (fileCharacters.length - 1) / exampleLength - 2;   //-2: for end index, and for partial example
        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            exampleStartOffsets.add(i * exampleLength);
        }
        Collections.shuffle(exampleStartOffsets, rng);
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    public int batch() {
        return miniBatchSize;
    }

    public int cursor() {
        return totalExamples() - exampleStartOffsets.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

}
```

////////////////////////////////////////////////////

# CustomLayerExample

Note that the custom layer JSON needs to be modified to point to the file in the unit tests: org.deeplearning4j.regressiontest.customlayer100a.CustomLayer

```java
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CustomLayerExample {

    public static void main(String[] args) throws IOException {
        DataType[] dtypes = new DataType[]{ DataType.DOUBLE, DataType.FLOAT, DataType.HALF };
        for(DataType dtype :dtypes)
            makeTest(dtype);
    }

    public static void makeTest(DataType dtype) throws IOException {
        Nd4j.setDefaultDataTypes(dtype, dtype);
        String dtypeName = dtype.toString().toLowerCase();
        int nIn = 5;
        int nOut = 8;

        //Let's create a network with our custom layer

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()

                .updater( new RmsProp(0.95))
                .weightInit(WeightInit.XAVIER)
                .l2(0.03)
                .list()
                .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(6).build())     //Standard DenseLayer
                .layer(1, new CustomLayer.Builder()
                        .activation(Activation.TANH)                                                    //Property inherited from FeedForwardLayer
                        .secondActivationFunction(Activation.SIGMOID)                                   //Custom property we defined for our layer
                        .nIn(6).nOut(7)                                                                 //nIn and nOut also inherited from FeedForwardLayer
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)                //Standard OutputLayer
                        .activation(Activation.SOFTMAX).nIn(7).nOut(nOut).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();

        net.save(new File("output/CustomLayerExample_100b4_" + dtypeName + ".bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{3, nIn});
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/CustomLayerExample_Input_100b4_" + dtypeName + ".bin")))) {
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input);
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/CustomLayerExample_Output_100b4_" + dtypeName + ".bin")))) {
            Nd4j.write(output, dos);
        }
    }

}
```

```java
import java.util.Collection;
import java.util.Map;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public class CustomLayer extends FeedForwardLayer {

    private IActivation secondActivationFunction;

    public CustomLayer() {
        //We need a no-arg constructor so we can deserialize the configuration from JSON or YAML format
        // Without this, you will likely get an exception like the following:
        //com.fasterxml.jackson.databind.JsonMappingException: No suitable constructor found for type [simple type, class org.deeplearning4j.examples.misc.customlayers.layer.CustomLayer]: can not instantiate from JSON object (missing default constructor or creator, or perhaps need to add/enable type information?)
    }

    private CustomLayer(Builder builder) {
        super(builder);
        this.secondActivationFunction = builder.secondActivationFunction;
    }

    public IActivation getSecondActivationFunction() {
        //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
        return secondActivationFunction;
    }

    public void setSecondActivationFunction(IActivation secondActivationFunction) {
        //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
        this.secondActivationFunction = secondActivationFunction;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDType) {
        //The instantiate method is how we go from the configuration class (i.e., this class) to the implementation class
        // (i.e., a CustomLayerImpl instance)
        //For the most part, it's the same for each type of layer

        CustomLayerImpl myCustomLayer = new CustomLayerImpl(conf, networkDType);
        myCustomLayer.setListeners(iterationListeners);             //Set the iteration listeners, if any
        myCustomLayer.setIndex(layerIndex);                         //Integer index of the layer

        //Parameter view array: In Deeplearning4j, the network parameters for the entire network (all layers) are
        // allocated in one big array. The relevant section of this parameter vector is extracted out for each layer,
        // (i.e., it's a "view" array in that it's a subset of a larger array)
        // This is a row vector, with length equal to the number of parameters in the layer
        myCustomLayer.setParamsViewArray(layerParamsView);

        //Initialize the layer parameters. For example,
        // Note that the entries in paramTable (2 entries here: a weight array of shape [nIn,nOut] and biases of shape [1,nOut]
        // are in turn a view of the 'layerParamsView' array.
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        myCustomLayer.setParamTable(paramTable);
        myCustomLayer.setConf(conf);
        return myCustomLayer;
    }

    @Override
    public ParamInitializer initializer() {
        //This method returns the parameter initializer for this type of layer
        //In this case, we can use the DefaultParamInitializer, which is the same one used for DenseLayer
        //For more complex layers, you may need to implement a custom parameter initializer
        //See the various parameter initializers here:
        //https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/params

        return DefaultParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //Memory report is used to estimate how much memory is required for the layer, for different configurations
        //If you don't need this functionality for your custom layer, you can return a LayerMemoryReport
        // with all 0s, or

        //This implementation: based on DenseLayer implementation
        InputType outputType = getOutputType(-1, inputType);

        long numParams = initializer().numParams(this);
        int updaterStateSize = (int)getIUpdater().stateSize(numParams);

        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if(getIDropout() != null){
            //Assume we dup the input for dropout
            trainSizeVariable += inputType.arrayElementsPerExample();
        }

        //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
        // which is modified in-place by activation function backprop
        // then we have 'epsilonNext' which is equivalent to input size
        trainSizeVariable += outputType.arrayElementsPerExample();

        return new LayerMemoryReport.Builder(layerName, CustomLayer.class, inputType, outputType)
            .standardMemory(numParams, updaterStateSize)
            .workingMemory(0, 0, trainSizeFixed, trainSizeVariable)     //No additional memory (beyond activations) for inference
            .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
            .build();
    }


    //Here's an implementation of a builder pattern, to allow us to easily configure the layer
    //Note that we are inheriting all of the FeedForwardLayer.Builder options: things like n
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        private IActivation secondActivationFunction;

        //This is an example of a custom property in the configuration

        /**
         * A custom property used in this custom layer example. See the CustomLayerExampleReadme.md for details
         *
         * @param secondActivationFunction Second activation function for the layer
         */
        public Builder secondActivationFunction(String secondActivationFunction) {
            return secondActivationFunction(Activation.fromString(secondActivationFunction));
        }

        /**
         * A custom property used in this custom layer example. See the CustomLayerExampleReadme.md for details
         *
         * @param secondActivationFunction Second activation function for the layer
         */
        public Builder secondActivationFunction(Activation secondActivationFunction){
            this.secondActivationFunction = secondActivationFunction.getActivationFunction();
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
        public CustomLayer build() {
            return new CustomLayer(this);
        }
    }

}
```

```java
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

public class CustomLayerImpl extends BaseLayer<CustomLayer> { //Generic parameter here: the configuration class type

    public CustomLayerImpl(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }



    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        /*
        The activate method is used for doing forward pass. Note that it relies on the pre-output method;
        essentially we are just applying the activation function (or, functions in this example).
        In this particular (contrived) example, we have TWO activation functions - one for the first half of the outputs
        and another for the second half.
         */

        INDArray output = preOutput(training, workspaceMgr);
        int columns = output.columns();

        INDArray firstHalf = output.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray secondHalf = output.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        IActivation activation1 = layerConf().getActivationFn();
        IActivation activation2 = ((CustomLayer) conf.getLayer()).getSecondActivationFunction();

        //IActivation function instances modify the activation functions in-place
        activation1.getActivation(firstHalf, training);
        activation2.getActivation(secondHalf, training);

        return output;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        /*
        The baockprop gradient method here is very similar to the BaseLayer backprop gradient implementation
        The only major difference is the two activation functions we have added in this example.
        Note that epsilon is dL/da - i.e., the derivative of the loss function with respect to the activations.
        It has the exact same shape as the activation arrays (i.e., the output of preOut and activate methods)
        This is NOT the 'delta' commonly used in the neural network literature; the delta is obtained from the
        epsilon ("epsilon" is dl4j's notation) by doing an element-wise product with the activation function derivative.
        Note the following:
        1. Is it very important that you use the gradientViews arrays for the results.
           Note the gradientViews.get(...) and the in-place operations here.
           This is because DL4J uses a single large array for the gradients for efficiency. Subsets of this array (views)
           are distributed to each of the layers for efficient backprop and memory management.
        2. The method returns two things, as a Pair:
           (a) a Gradient object (essentially a Map<String,INDArray> of the gradients for each parameter (again, these
               are views of the full network gradient array)
           (b) an INDArray. This INDArray is the 'epsilon' to pass to the layer below. i.e., it is the gradient with
               respect to the input to this layer
        */

        INDArray activationDerivative = preOutput(true, workspaceMgr);
        int columns = activationDerivative.columns();

        INDArray firstHalf = activationDerivative.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray secondHalf = activationDerivative.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        INDArray epsilonFirstHalf = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, columns / 2));
        INDArray epsilonSecondHalf = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(columns / 2, columns));

        IActivation activation1 = layerConf().getActivationFn();
        IActivation activation2 = ((CustomLayer) conf.getLayer()).getSecondActivationFunction();

        //IActivation backprop method modifies the 'firstHalf' and 'secondHalf' arrays in-place, to contain dL/dz
        activation1.backprop(firstHalf, epsilonFirstHalf);
        activation2.backprop(secondHalf, epsilonSecondHalf);

        //The remaining code for this method: just copy & pasted from BaseLayer.backpropGradient
//        INDArray delta = epsilon.muli(activationDerivative);
        if (maskArray != null) {
            activationDerivative.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();

        INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);    //f order
        Nd4j.gemm(input, activationDerivative, weightGrad, true, false, 1.0, 0.0);
        INDArray biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
        biasGrad.assign(activationDerivative.sum(0));  //TODO: do this without the assign

        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGrad);

        INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(activationDerivative.transpose()).transpose();

        return new Pair<>(ret, workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilonNext));
    }

}
```




////////////////////////////////////////////////////

# VaeMnistAnomaly

```java
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

public class VaeMNISTAnomaly {

    public static void main(String[] args) throws IOException {
        int minibatchSize = 128;
        int rngSeed = 12345;
        int nEpochs = 5;                    //Total number of training epochs
        int reconstructionNumSamples = 16;  //Reconstruction probabilities are estimated using Monte-Carlo techniques; see An & Cho for details

        //MNIST data for training
        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed);

        //Neural net configuration
        Nd4j.getRandom().setSeed(rngSeed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .updater(new Adam(1e-3))
            .weightInit(WeightInit.XAVIER)
            .l2(1e-4)
            .list()
            .layer(new VariationalAutoencoder.Builder()
                .activation(Activation.LEAKYRELU)
                .encoderLayerSizes(256, 256)                    //2 encoder layers, each of size 256
                .decoderLayerSizes(256, 256)                    //2 decoder layers, each of size 256
                .pzxActivationFunction(Activation.IDENTITY)     //p(z|data) activation function
                //Bernoulli reconstruction distribution + sigmoid activation - for modelling binary data (or data in range 0 to 1)
                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                .nIn(28 * 28)                                   //Input size: 28x28
                .nOut(32)                                       //Size of the latent variable space: p(z|x) - 32 values
                .build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.save(new File("output/VaeMNISTAnomaly_100b4.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(3, 28*28);
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/VaeMNISTAnomaly_Input_100b4.bin")))) {
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input);
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/VaeMNISTAnomaly_Output_100b4.bin")))) {
            Nd4j.write(output, dos);
        }
    }
}
```


////////////////////////////////////////////////////

# HouseNumberDetection

```java
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.Random;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HouseNumberDetection {
    private static final Logger log = LoggerFactory.getLogger(HouseNumberDetection.class);

    public static void main(String[] args) throws java.lang.Exception {

        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        // number classes (digits) for the SVHN datasets
        int nClasses = 10;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 1.0;
        double[][] priorBoxes = {{2, 5}, {2.5, 6}, {3, 7}, {3.5, 8}, {4, 9}};
        double detectionThreshold = 0.5;

        // parameters for the training phase
        int batchSize = 10;
        int nEpochs = 20;
        double learningRate = 1e-4;
        double lrMomentum = 0.9;

        int seed = 123;
        Random rng = new Random(seed);

//        SvhnDataFetcher fetcher = new SvhnDataFetcher();
//        File trainDir = fetcher.getDataSetPath(DataSetType.TRAIN);
//        File testDir = fetcher.getDataSetPath(DataSetType.TEST);


        log.info("Load data...");

//        FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);
//        FileSplit testData = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, rng);

        ObjectDetectionRecordReader recordReaderTrain = null; //new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth, new SvhnLabelProvider(trainDir));
//        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = null; //new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth, new SvhnLabelProvider(testDir));
//        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));


        ComputationGraph model;
        String modelFilename = "model.zip";

        if (new File(modelFilename).exists()) {
            log.info("Load model...");

            model = ModelSerializer.restoreComputationGraph(modelFilename);
        } else {
            log.info("Build model...");

            ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
            INDArray priors = Nd4j.create(priorBoxes);

            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    .updater(new Adam.Builder().learningRate(learningRate).build())
                    //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                    .l2(0.00001)
                    .activation(Activation.IDENTITY)
                    .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                    .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
                    .build();

            model = new TransferLearning.GraphBuilder(pretrained)
                    .fineTuneConfiguration(fineTuneConf)
                    .removeVertexKeepConnections("conv2d_9")
                    .removeVertexKeepConnections("outputs")
                    .addLayer("convolution2d_9",
                            new ConvolutionLayer.Builder(1, 1)
                                    .nIn(1024)
                                    .nOut(nBoxes * (5 + nClasses))
                                    .stride(1, 1)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .weightInit(WeightInit.XAVIER)
                                    .activation(Activation.IDENTITY)
                                    .build(),
                            "leaky_re_lu_8")
                    .addLayer("outputs",
                            new Yolo2OutputLayer.Builder()
                                    .lambbaNoObj(lambdaNoObj)
                                    .lambdaCoord(lambdaCoord)
                                    .boundingBoxPriors(priors)
                                    .build(),
                            "convolution2d_9")
                    .setOutputs("outputs")
                    .build();

            model.save(new File("output/HouseNumberDetection_100b4.bin"));
            Nd4j.getRandom().setSeed(12345);
            INDArray input = Nd4j.rand(new int[]{3, 3, 416, 416});
            try (DataOutputStream dos = new DataOutputStream(
                    new FileOutputStream(new File("output/HouseNumberDetection_Input_100b4.bin")))) {
                Nd4j.write(input, dos);
            }
            INDArray output = model.outputSingle(input);
            try (DataOutputStream dos = new DataOutputStream(
                    new FileOutputStream(new File("output/HouseNumberDetection_Output_100b4.bin")))) {
                Nd4j.write(output, dos);
            }
        }
    }
}
```
Requires `org.deeplearning4j:deeplearning4j-zoo`.

////////////////////////////////////////////////////

# Synthetic CNN

```java
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.DepthwiseConvolution2D;
import org.deeplearning4j.nn.conf.layers.Pooling2D;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class SyntheticCNN {

    public static void main(String[] args) throws Exception {
        int height = 28;    // height of the picture in px
        int width = 28;     // width of the picture in px
        int channels = 1;   // single channel for grayscale images

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.005))
                .list()
                .layer(new Convolution2D.Builder()
                        .kernelSize(3, 3)
                        .stride(2, 1)
                        .nOut(4).nIn(channels)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SeparableConvolution2D.Builder()
                        .kernelSize(3, 3)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(8)
                        .activation(Activation.RELU)
                        .build())
                .layer(new Pooling2D.Builder().kernelSize(3, 3).poolingType(PoolingType.MAX).build())
                .layer(new ZeroPaddingLayer(4, 4))
                .layer(new Upsampling2D.Builder().size(3).build())
                .layer(new DepthwiseConvolution2D.Builder()
                        .kernelSize(3, 3)
                        .depthMultiplier(2)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2, 2).poolingType(PoolingType.MAX).build())
                .layer(new Cropping2D.Builder(3, 2).build())
                .layer(new Convolution2D.Builder().kernelSize(4, 4).nOut(4).build())
                .layer(new CnnLossLayer.Builder().lossFunction(LossFunction.MEAN_ABSOLUTE_ERROR).build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.save(new File("output/SyntheticCNN_100b4.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{4, 1, 28, 28});
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/SyntheticCNN_Input_100b4.bin")))) {
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input);
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/SyntheticCNN_Output_100b4.bin")))) {
            Nd4j.write(output, dos);
        }
    }
}
```

////////////////////////////////////////////////////

# Synthetic Bidirectional RNN Graph

```java
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class SyntheticBidirectionalRNNGraph {

    public static void main(String[] args) throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(0.0001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.005))
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.recurrent(10, 10))
                .addLayer("rnn1",
                        new Bidirectional(new LSTM.Builder()
                                .nOut(16)
                                .activation(Activation.RELU)
                                .build()), "input")
                .addLayer("rnn2",
                        new Bidirectional(new SimpleRnn.Builder()
                                .nOut(16)
                                .activation(Activation.RELU)
                                .build()), "input")
                .addVertex("concat", new MergeVertex(), "rnn1", "rnn2")
                .addLayer("pooling", new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.MAX)
                        .poolingDimensions(2)
                        .collapseDimensions(true)
                        .build(), "concat")
                .addLayer("out", new OutputLayer.Builder().nOut(3).lossFunction(LossFunction.MCXENT).build(), "pooling")
                .setOutputs("out")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        net.save(new File("output/SyntheticBidirectionalRNNGraph_100b4.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{4, 10, 10});
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/SyntheticBidirectionalRNNGraph_Input_100b4.bin")))) {
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input)[0];
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/SyntheticBidirectionalRNNGraph_Output_100b4.bin")))) {
            Nd4j.write(output, dos);
        }
    }
}
```
