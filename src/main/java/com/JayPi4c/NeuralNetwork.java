package com.JayPi4c;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;

/**
 * @author JayPi4c
 * @version 1.2.0
 */
public class NeuralNetwork implements Serializable {

	private static final long serialVersionUID = 5795625580326323029L;
	protected static final String err_Message = "An error has occured. Please contact jonas4c@freenet.de to get help on this problem. Please consider to add the following errorcode for debugging purposes: ";

	public static final ActivationFunction sigmoid = new ActivationFunction() {

		@Override
		public double deactivate(double y) {
			return y * (1 - y);
		}

		@Override
		public double activate(double x) {
			return (1 / (1 + Math.pow(Math.E, -x)));
		}
	};
	public static final ActivationFunction tanh = new ActivationFunction() {

		@Override
		public double deactivate(double y) {
			return 1 - (y * y);
		}

		@Override
		public double activate(double x) {
			return Math.tanh(x);
		}
	};

	protected ActivationFunction actFunc = sigmoid;

	/**
	 * This array includes the number of nodes for each layer<br>
	 * - layers[0] -> number of nodes in input layer<br>
	 * - layers[nodes.length-1] -> number of nodes in output layer<br>
	 * - layers[n] -> number of nodes in nth hidden layer
	 */
	protected int[] layers;
	protected double learningrate;

	/**
	 * This array includes the weight matrices for each layer<br>
	 * - weights[0] -> weights for input hidden#1 - weights[weights.length-1]<br>
	 * - weights for hidden->output
	 */
	protected Matrix[] weights;

	/**
	 * This array includes the bias matrices for each layer<br>
	 * -> biases[0] -> biases for layer hidden#1<br>
	 * -> biases[biases.length-1] -> biases for output layer
	 */
	protected Matrix[] biases;

	/**
	 * Erstellt ein Neuronales Netz mit der jeweiligen Anzahl an Input-, Hidden- und
	 * Outputnodes. Dieses Neuronale Netz kann mit
	 * {@link #train(double[][], double[][])} und {@link #query(double[][])} benutzt
	 * werden.
	 * 
	 * @param inputnodes
	 * @param hiddennodes
	 * @param hiddennodes
	 * @param learningrate
	 * @see #train(double[][], double[][]), {@link #train(Matrix, Matrix)},
	 *      {@link #query(double[][])}, {@link #query(Matrix)}
	 * @since 1.0.0
	 */
	public NeuralNetwork(double learningrate, int inputnodes, int outputnodes, int... hiddennodes) {
		this.learningrate = learningrate;

		this.layers = new int[hiddennodes.length + 2];

		// add input nodes to layers
		if (inputnodes < 1)
			throw new IllegalArgumentException("Inputnodes must at least be one!");
		else
			this.layers[0] = inputnodes;

		// add output nodes to layers
		if (outputnodes < 1)
			throw new IllegalArgumentException("Outputnodes must at least be one!");
		else
			this.layers[layers.length - 1] = outputnodes;

		// add hidden nodes to layers
		if (hiddennodes == null || hiddennodes.length == 0)
			throw new IllegalArgumentException("At least one hidden layer must be provided!");
		for (int i = 0; i < hiddennodes.length; i++)
			if (hiddennodes[i] < 1)
				throw new IllegalArgumentException(
						"All hidden layers must at least have one neuron, which is not true for layer #" + (i + 1));
			else
				layers[i + 1] = hiddennodes[i];

		weights = new Matrix[layers.length - 1];
		biases = new Matrix[layers.length - 1];
		for (int i = 1; i < layers.length; i++) {
			weights[i - 1] = new Matrix(layers[i], layers[i - 1]).randomize(-0.5, 0.5);
			biases[i - 1] = new Matrix(layers[i], 1).randomize(-0.5, 0.5);
		}

	}

	// ****************************************************************************************************************//
	/**
	 * Durch die gegebenen Eingaben l&aumlsst sich mit den Gewichten eine Ausgabe
	 * ermitteln, die die Sch&aumltzung des Neuronalen Netzes wider spiegeln.
	 * 
	 * @param inputs_list
	 * @return die Sch&aumltzung des Neuronalen Netzes.
	 * @since 1.0.0
	 */
	public Matrix query(Matrix inputs_list) {

		Matrix matrix = Matrix.transpose(inputs_list);
		for (int i = 0; i < weights.length; i++) {
			matrix = Matrix.dot(weights[i], matrix);
			matrix.add(biases[i]);
			matrix.map(d -> actFunc.activate(d));
		}
		return matrix;
	}

	/**
	 * Durch die gegebenen Eingaben l&aumlsst sich mit den Gewichten eine Ausgabe
	 * ermitteln, die die Sch&aumltzung des Neuronalen Netzes wider spiegeln.
	 * 
	 * @param inputs_list
	 * @return die Sch&aumltzung des Neuronalen Netzes.
	 * @since 1.0.0
	 */
	public Matrix query(double inputs_list[][]) {
		return this.query(new Matrix(inputs_list));
	}

	// ****************************************************************************************************************//
	/**
	 * Durch angabe der Eingabe und der gew&uumlnschten Ausgabe k&oumlnnen die
	 * Gewichte des Neuronalen Netzes angepasst werden.
	 * 
	 * @param inputs_list  die Eingaben in das Neuronale Netz
	 * @param targets_list die Ausgabe des Neuronalen Netzes
	 * @throws Exception
	 * @since 1.0.0
	 */
	public void train(Matrix inputs_list, Matrix targets_list) {

		Matrix results[] = new Matrix[weights.length + 1];
		results[0] = Matrix.transpose(inputs_list);
		for (int i = 0; i < weights.length; i++) {
			results[i + 1] = Matrix.dot(weights[i], results[i]);
			results[i + 1].add(biases[i]);
			results[i + 1].map(d -> actFunc.activate(d));
		}

		Matrix error = Matrix.sub(targets_list, results[results.length - 1]);
		Matrix gradients = results[results.length - 1].map(d -> actFunc.deactivate(d));
		gradients.mult(error);
		gradients.mult(learningrate);

		// calculate deltas
		Matrix prev_T = Matrix.transpose(results[results.length - 2]);
		Matrix weight_deltas = Matrix.dot(gradients, prev_T);

		// Adjust the weights by deltas
		weights[results.length - 2].add(weight_deltas);
		// Adjust the bias by its deltas (which is just the gradients)
		biases[results.length - 2].add(gradients);

		for (int i = results.length - 2; i >= 1; i--) {
			// calculate error
			error = Matrix.dot(Matrix.transpose(weights[i]), error);

			gradients = results[i].map(d -> actFunc.deactivate(d));
			gradients.mult(error);
			gradients.mult(learningrate);

			// calculate deltas
			prev_T = Matrix.transpose(results[i - 1]);
			weight_deltas = Matrix.dot(gradients, prev_T);

			// Adjust the weights by deltas
			weights[i - 1].add(weight_deltas);
			// Adjust the bias by its deltas (which is just the gradients)
			biases[i - 1].add(gradients);
		}
	}

	/**
	 * Durch angabe der Eingabe und der gew&uumlnschten Ausgabe k&oumlnnen die
	 * Gewichte des Neuronalen Netzes angepasst werden.
	 * 
	 * @param inputs_list  die Eingaben in das Neuronale Netz
	 * @param targets_list die Ausgabe des Neuronalen Netzes
	 * @since 1.0.0
	 */
	public void train(double inputs_list[][], double targets_list[][]) {
		this.train(new Matrix(inputs_list), new Matrix(targets_list));
	}

	// ****************************************************************************************************************//

	/**
	 * 
	 * Mit dieser Funktion l&aumlsst sich das Objekt des Neuronalen Netzes in einer
	 * Datei speichern, sodass auch nach beenden des Programms der Fortschritt bzw.
	 * der Zustand des Netzes gespeichert bleiben kann. Die ndene Datei kann
	 * anschlie&szligend bzw. bei einem Neustart des Programms mit
	 * {@link #deserialize(File)} wieder eingelesen werden.
	 * 
	 * @param nn
	 * @throws IOException
	 * @see #deserialize(File)
	 * @since 1.0.0
	 */
	public void serialize() throws IOException {
		String absolutePath = new File(".").getAbsolutePath();
		File file = new File(absolutePath);
		absolutePath = file.getParentFile().toString();
		FileOutputStream fos = new FileOutputStream(new File(absolutePath + "/NeuralNetwork.nn"));
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(this);
		oos.close();
	}

	/**
	 * 
	 * Mit dieser Funktion l&aumlsst sich ein Objekt des Neuronalen Netzes in einer
	 * Datei speichern, sodass auch nach beenden des Programms der Fortschritt bzw.
	 * der Zustand des Netzes gespeichert bleiben kann. Die entstandene Datei kann
	 * anschlie&szligend bzw. bei einem neustart des Programms mit
	 * {@link #deserialize(File)} wieder eingelesen werden.
	 * 
	 * @param nn
	 * @throws IOException
	 * @see #deserialize(File)
	 * @since 1.0.0
	 */
	public static void serialize(NeuralNetwork nn) throws IOException {
		String absolutePath = new File(".").getAbsolutePath();
		File file = new File(absolutePath);
		absolutePath = file.getParentFile().toString();
		FileOutputStream fos = new FileOutputStream(new File(absolutePath + "/NeuralNetwork.nn"));
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(nn);
		oos.close();
	}

	/**
	 * Mit dieser Funktion l&aumlsst sich ein Neuronales Netz Objekt aus einer Datei
	 * erstellen, die mit {@link #serialize()} erstellt wurde.
	 * 
	 * @param f Ein File Object, welches die Daten des Neuronalen Netzes beinhaltet
	 * @return gibt ein Neuronales Netz zur&uumlck
	 * @throws IOException
	 * @throws ClassNotFoundException
	 * @see #serialize()
	 * @since 1.0.0
	 */
	public static NeuralNetwork deserialize(File f) throws IOException, ClassNotFoundException {
		FileInputStream fis = new FileInputStream(f);
		ObjectInputStream ois = new ObjectInputStream(fis);

		// NeuralNetwork output = null;
		NeuralNetwork output = (NeuralNetwork) ois.readObject();
		ois.close();
		return output;
	}

	/**
	 * creates a Buffered Image representing the neural networks nodes and weights
	 * with the specified background.
	 * 
	 * @param background the specified color for the background
	 * @param width      the width of the image
	 * @param height     the height of the image
	 * @return
	 * @since 1.1.0
	 */
	public BufferedImage getSchemeImage(Color background, int width, int height) {
		int radius = 5, diameter = radius * 2;
		BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
		Graphics2D graphics = (Graphics2D) img.getGraphics();
		if (background != null) {
			graphics.setColor(background);
			graphics.fillRect(0, 0, width, height);
		}
		// top and bottom will have 10 pixels spare to the border
		int y_min = (int) (0.02 * height);
		int y_max = height - y_min;
		int x_min = (int) (0.02 * width);
		int x_max = width - x_min;

		double x = (x_max - x_min) / (double) (layers.length - 1);

		// draw the weights
		for (int layer = 0; layer < weights.length; layer++) {
			int x_left = (int) (x_min + layer * x);
			int x_right = (int) (x_left + x);
			double y_spacer_left = (y_max - y_min) / (double) (layers[layer] + 1);
			double y_spacer_right = (y_max - y_min) / (double) (layers[layer + 1] + 1);
			for (int left_nodes = 0; left_nodes < layers[layer]; left_nodes++) {
				for (int right_nodes = 0; right_nodes < layers[layer + 1]; right_nodes++) {
					float val = (float) weights[layer].data[right_nodes][left_nodes];
					float abs_val = Math.min(Math.abs(val), 1);
					graphics.setColor(new Color(val < 0 ? abs_val : 0f, val > 0 ? abs_val : 0f, 0f, abs_val));
					graphics.drawLine(x_left, (int) (y_min + (left_nodes + 1) * y_spacer_left), x_right,
							(int) (y_min + (right_nodes + 1) * y_spacer_right));
				}
			}
		}

		// draw nodes
		// set node color according to bias
		graphics.setColor(Color.BLUE);
		for (int i = 0; i < layers.length; i++) {
			double y = (y_max - y_min) / (double) (layers[i] + 1);
			for (int node = 0; node < layers[i]; node++) {
				if (i > 0) {
					float val = (float) biases[i - 1].data[node][0];
					float abs_val = Math.min(Math.abs(val), 1);
					graphics.setColor(new Color(val < 0 ? abs_val : 0f, val > 0 ? abs_val : 0f, 0f, abs_val));
				}
				graphics.fillOval((int) (x_min + i * x - radius), (int) (y_min + (node + 1) * y - radius), diameter,
						diameter);

			}
		}

		return img;
	}

	/**
	 * 
	 * @return a scheme image with a size of 640x480 and transparent background
	 * @since 1.1.0
	 */
	public BufferedImage getSchemeImage() {
		return getSchemeImage(null, 640, 480);
	}

	/**
	 * Creates a Scheme Image of the Neural Network with the specified size
	 * 
	 * @param width  width of the image
	 * @param height height of the image
	 * @return a BufferedImage with the specified size and a transparent Background
	 * @since 1.1.0
	 */
	public BufferedImage getSchemeImage(int width, int height) {
		return getSchemeImage(null, width, height);
	}

	/**
	 * 
	 * @param actFunc
	 */
	public void setActivationFunction(ActivationFunction actFunc) {
		this.actFunc = actFunc;
	}

	/**
	 * 
	 * @return Gibt eine Kopie des neuronalen Netzes zur&uumlck, welche in die exakt
	 *         gleichen Werte hat, wie das originale Netz, aber keinen Bezug hat.
	 * @since 1.0.0
	 */
	public NeuralNetwork copy() {
		NeuralNetwork output = null;
		output = new NeuralNetwork(this.learningrate, this.layers[0], this.layers[layers.length - 1],
				Arrays.copyOfRange(layers, 1, layers.length - 1));
		for (int i = 0; i < weights.length; i++)
			output.weights[i] = weights[i].copy();

		for (int i = 0; i < biases.length; i++)
			output.biases[i] = biases[i].copy();

		return output;
	}

}