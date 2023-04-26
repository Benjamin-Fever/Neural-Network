import java.io.IOException;
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
        double output = 1 / (1 + Math.exp(-input)); //TODO! ===XXX===
        return output;

    }

    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output. ===XXX===
            double weighted_sum = calculate_weighted_sums(num_inputs, hidden_layer_weights, inputs, i);
            double output = sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            // TODO! Calculate the weighted sum, and then compute the final output. ===XXX===
            double weighted_sum = calculate_weighted_sums(num_hidden, output_layer_weights, hidden_layer_outputs, i);
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

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
                                                 double[] output_layer_outputs, int desired_output) {

        double[] output_layer_betas = new double[num_outputs];
        double[] desired_outputs = desired_output_convert(desired_output);
        // TODO! Calculate output layer betas.
        for (int z = 0; z < num_outputs; z++){
            output_layer_betas[z] = desired_outputs[z] - output_layer_outputs[z];
        }

        System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

        double[] hidden_layer_betas = new double[num_hidden];
        // TODO! Calculate hidden layer betas.
        for (int j = 0; j < num_hidden; j++){
            hidden_layer_betas[j] = 0;
            for (int k = 0; k < num_hidden; k++) {
                hidden_layer_betas[j] += hidden_layer_weights[j][k] * hidden_layer_outputs[k] *
                        (1-hidden_layer_outputs[k]) * output_layer_betas[k];
            }
        }
        System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

        // This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
        // TODO! Calculate output layer weight changes.
        // delta_output_layer_weights[j][k] = output_layer_betas[k] * hidden_layer_outputs[j] * learning_rate;
        //        }

        // This is a IxH array (I inputs, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        // TODO! Calculate hidden layer weight changes.

        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights};
    }

    public double[] desired_output_convert(int desired_output){
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

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights) {
        // TODO! Update the weights
        System.out.println("Placeholder");
    }

    public void train(double[][] instances, int[] desired_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("epoch = " + epoch);
            int[] predictions = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);
                int predicted_class = -1; // TODO!
                predictions[i] = predicted_class;

                //We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1]);
            }

            // Print new weights
            System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
            System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            // TODO: Print accuracy achieved over this epoch
            double acc = Double.NaN;
            System.out.println("acc = " + acc);
        }
    }

    public int[] predict(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            int predicted_class = -1;  // TODO !Should be 0, 1, or 2.
            predictions[i] = predicted_class;
        }
        return predictions;
    }

}
