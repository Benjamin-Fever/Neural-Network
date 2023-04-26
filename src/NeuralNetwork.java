import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;

    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;

    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.learning_rate = learning_rate;
    }


    //Calculate neuron activation for an input
    public double sigmoid(double input) {
        double output = 1 / (1 + Math.exp(-input));
        return output;

    }

    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            double weighted_sum = calculate_weighted_sums(num_inputs, hidden_layer_weights, inputs, i);
            weighted_sum += hidden_layer_weights[num_inputs][i];
            double output = sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            double weighted_sum = calculate_weighted_sums(num_hidden, output_layer_weights, hidden_layer_outputs, i);
            weighted_sum += output_layer_weights[num_hidden][i];
            double output = sigmoid(weighted_sum);
            output_layer_outputs[i] = output;
        }
        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    public double calculate_weighted_sums(int iterations, double[][] weights, double[] inputs, int i){
        double weighted_sum = 0;
        for (int n = 0; n < iterations; n++){
            double input = inputs[n];
            weighted_sum += weights[n][i] * input;
        }
        return weighted_sum;
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs, double[] output_layer_outputs, int desired_output) {

        double[] output_layer_betas = new double[num_outputs];
        double[] desired_outputs = output_convert(desired_output);
        for (int z = 0; z < num_outputs; z++){
            output_layer_betas[z] = desired_outputs[z] - output_layer_outputs[z];
        }

        //System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

        double[] hidden_layer_betas = new double[num_hidden];
        for (int j = 0; j < num_hidden; j++){
            hidden_layer_betas[j] = 0;
            for (int k = 0; k < num_hidden; k++) {
                double weight = hidden_layer_weights[j][k];
                double beta = output_layer_betas[k];
                double output = output_layer_outputs[k];
                hidden_layer_betas[j] += weight * beta * output * (1 - output);
            }
        }
        //System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

        // This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
        for (int i = 0; i < num_outputs; i++){
            double output = output_layer_outputs[i];
            double beta = output_layer_betas[i];

            for (int j = 0; j < num_hidden; j++){
                double hidden_output = hidden_layer_outputs[j];
                delta_output_layer_weights[j][i] = learning_rate * beta * hidden_output * output * (1 - output);
            }
        }
        

        // This is a IxH array (I inputs, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        for (int i = 0; i < num_hidden; i++){
            double output = hidden_layer_outputs[i];
            double beta = hidden_layer_betas[i];
            for (int j = 0; j < num_inputs; j++){
                double input = inputs[j];
                delta_hidden_layer_weights[j][i] = learning_rate * beta * input * output * (1 - output);
            }
        }

        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights};
    }

    public double[] output_convert(int desired_output){
        switch (desired_output) {
            case 0:
                return new double[]{1,0,0};
            case 1:
                return new double[]{0,1,0};
            case 2:
                return new double[]{0,0,1};
            default:
                System.out.println("Error: desired output is not 0, 1 or 2");
                break;
        }
        return null;
    }

    public int output_convert(double[] output){
        int value = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[value]) {
                value = i;
            }
        }
        return value;
    }

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights) {
        for (int i = 0; i < num_hidden; i++){
            for (int j = 0; j < num_outputs; j++){
                output_layer_weights[i][j] += delta_output_layer_weights[i][j];
            }
        }
        
        for (int i = 0; i < num_inputs; i++){
            for (int j = 0; j < num_hidden; j++){
                hidden_layer_weights[i][j] += delta_hidden_layer_weights[i][j];
            }
        }
    }

    public void train(double[][] instances, int[] desired_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("epoch = " + epoch);
            int[] predictions = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);
                int predicted_class = output_convert(outputs[1]);
                predictions[i] = predicted_class;

                //We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1]);
            }

            // Print new weights
            System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
            System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            System.out.println("acc = " + accuracy_calc(desired_outputs, predictions));
        }
    }

    public double accuracy_calc(int[] desired, int[] predictions){
        int count = 0;
        for(int i = 0; i < predictions.length; i++){
            if(predictions[i] == desired[i]){
                count++;
            }
        }
        double calc = (double)count / (double)desired.length;
        return calc;
    }

    public int[] predict(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            int predicted_class = output_convert(outputs[1]);
            predictions[i] = predicted_class;
        }
        return predictions;
    }

    public int[] predict(double[][] instances, Boolean print) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            System.out.println("Feedforward Outputs: [" + outputs[1][0] + ", " + outputs[1][1] + ", " + outputs[1][2] + "]");
            int predicted_class = output_convert(outputs[1]);
            predictions[i] = predicted_class;
        }
        return predictions;
    }

}
