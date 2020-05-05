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

/**
 * @author JayPi4c
 * @version 1.1.0
 */
public class NeuralNetwork implements Serializable {

	private static final long serialVersionUID = 5795625580326323029L;
	protected static final String err_Message = "An error has occured. Please contact jonas4c@freenet.de to get help on this problem. Please consider to add the following errorcode for debugging purposes: ";

	protected int inputnodes, outputnodes;
	protected int[] hiddennodes;
	protected double learningrate;

	protected Matrix wih, who;
	protected Matrix[] whh;

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

		this.inputnodes = inputnodes;
		if (inputnodes < 1)
			throw new IllegalArgumentException("Inputnodes must at least be one!");
		this.outputnodes = outputnodes;
		if (outputnodes < 1)
			throw new IllegalArgumentException("Outputnodes must at least be one!");
		this.hiddennodes = hiddennodes;
		if (hiddennodes == null || hiddennodes.length == 0)
			throw new IllegalArgumentException("At least one hidden layer must be provided!");
		for (int i = 0; i < hiddennodes.length; i++)
			if (hiddennodes[i] < 1)
				throw new IllegalArgumentException(
						"All hidden layers must at least have one neuron, which is not true for layer #" + (i + 1));

		this.wih = new Matrix(this.hiddennodes[0], this.inputnodes).randomize(-0.5, 0.5);

		this.whh = new Matrix[this.hiddennodes.length - 1];
		for (int i = 0; i < whh.length; i++)
			whh[i] = new Matrix(this.hiddennodes[i + 1], this.hiddennodes[i]).randomize(-0.5, 0.5);

		this.who = new Matrix(this.outputnodes, this.hiddennodes[this.hiddennodes.length - 1]).randomize(-0.5, 0.5);

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
		Matrix final_outputs = null;
		try {
			Matrix inputs = Matrix.transpose(inputs_list);
			Matrix hidden_inputs[] = new Matrix[hiddennodes.length];
			Matrix hidden_outputs[] = new Matrix[hiddennodes.length];

			hidden_inputs[0] = Matrix.dot(this.wih, inputs);
			hidden_outputs[0] = hidden_inputs[0].map(d -> sigmoid(d));

			for (int i = 1; i < hiddennodes.length; i++) {
				hidden_inputs[i] = Matrix.dot(this.whh[i - 1], hidden_outputs[i - 1]);
				hidden_outputs[i] = hidden_inputs[i].map(d -> sigmoid(d));
			}

			Matrix final_inputs = Matrix.dot(this.who, hidden_outputs[hiddennodes.length - 1]);
			final_outputs = final_inputs.map(d -> sigmoid(d));
		} catch (Exception e) {
			System.out.println(err_Message);
			System.err.println(e);
		}
		return final_outputs;
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

		Matrix inputs = Matrix.transpose(inputs_list);
		Matrix targets = Matrix.transpose(targets_list);

		Matrix hidden_inputs[] = new Matrix[hiddennodes.length];
		Matrix hidden_outputs[] = new Matrix[hiddennodes.length];

		hidden_inputs[0] = Matrix.dot(this.wih, inputs);
		hidden_outputs[0] = hidden_inputs[0].map(d -> sigmoid(d));

		for (int i = 1; i < hiddennodes.length; i++) {
			hidden_inputs[i] = Matrix.dot(this.whh[i - 1], hidden_outputs[i - 1]);
			hidden_outputs[i] = hidden_inputs[i].map(d -> sigmoid(d));
		}

		Matrix final_inputs = Matrix.dot(this.who, hidden_outputs[hiddennodes.length - 1]);
		Matrix final_outputs = final_inputs.map(d -> sigmoid(d));

		// Matrix inputs = Matrix.transpose(inputs_list);
		// Matrix targets = Matrix.transpose(targets_list);

		// Matrix hidden_inputs = Matrix.dot(this.wih, inputs);
		// Matrix hidden_outputs = hidden_inputs.map(d -> sigmoid(d));
		// Matrix final_inputs = Matrix.dot(this.who, hidden_outputs);
		// Matrix final_outputs = final_inputs.map(d -> sigmoid(d));

		Matrix output_errors = Matrix.sub(targets, final_outputs);

		Matrix hidden_errors[] = new Matrix[hiddennodes.length];

		hidden_errors[hiddennodes.length - 1] = Matrix.dot(Matrix.transpose(this.who), output_errors);

		for (int i = hiddennodes.length - 2; i >= 0; i--) {
			hidden_errors[i] = Matrix.dot(Matrix.transpose(whh[i]), hidden_errors[i + 1]);
		}

		this.who.add(Matrix
				.mult(Matrix.dot(Matrix.mult(output_errors, Matrix.mult(final_outputs, Matrix.sub(1, final_outputs))),
						Matrix.transpose(hidden_outputs[hiddennodes.length - 1])), this.learningrate));

		for (int i = hiddennodes.length - 1; i >= 1; i--) {
			this.whh[i - 1].add(Matrix.mult(Matrix.dot(
					Matrix.mult(hidden_errors[i], Matrix.mult(hidden_outputs[i], Matrix.sub(1, hidden_outputs[i]))),
					Matrix.transpose(hidden_outputs[i - 1])), this.learningrate));
		}

		this.wih.add(Matrix.mult(Matrix.dot(
				Matrix.mult(hidden_errors[0], Matrix.mult(hidden_outputs[0], Matrix.sub(1, hidden_outputs[0]))),
				Matrix.transpose(inputs)), this.learningrate));

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
	 * Ermittelt zu einem gegebenen x-Wert den passenden y-Wert der
	 * Sigmoid-Funktion.
	 * 
	 * @param x
	 * @return Der Wert der Sigmoid-Funktion f&uumlr das angegebene x
	 * @since 1.0.0
	 */
	public double sigmoid(double x) {
		return (1 / (1 + Math.pow(Math.E, -x)));
	}

	/**
	 * 
	 * Mit dieser Funktion l&aumlsst sich das Objekt des Neuronalen Netzes in einer
	 * Datei speichern, sodass auch nach beenden des Programms der Fortschritt bzw.
	 * der Zustand des Netzes gespeichert bleiben kann. Die entstandene Datei kann
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

		// this.wih.print();

		// draw the weights
		// input to hidden
		double y_input = (y_max - y_min) / (double) (inputnodes + 1);
		for (int i = 0; i < inputnodes; i++) {
			double y_hidden = (y_max - y_min) / (double) (hiddennodes[0] + 1);
			double x_hidden = (x_max - x_min) / (double) (hiddennodes.length + 1);

			for (int j = 0; j < hiddennodes[0]; j++) {
				float val = (float) this.wih.data[j][i];
				float abs_val = Math.min(Math.abs(val), 1);
				graphics.setColor(new Color(val < 0 ? abs_val : 0f, val > 0 ? abs_val : 0f, 0f, abs_val));
				graphics.drawLine(x_min, (int) (y_min + (i + 1) * y_input), (int) (x_min + x_hidden),
						(int) (y_min + (j + 1) * y_hidden));
			}
		}

		// hidden to hidden
		double x_spacer = (x_max - x_min) / (double) (hiddennodes.length + 1);
		for (int i = 0; i < whh.length; i++) {
			double x_input = x_min + (i + 1) * x_spacer;
			double x_output = x_input + x_spacer;
			double y_spacer_input = (y_max - y_min) / (double) (hiddennodes[i] + 1);
			double y_spacer_output = (y_max - y_min) / (double) (hiddennodes[i + 1] + 1);
			for (int node = 0; node < hiddennodes[i]; node++) {
				for (int nextNode = 0; nextNode < hiddennodes[i + 1]; nextNode++) {
					float val = (float) this.whh[i].data[nextNode][node];
					float abs_val = Math.min(Math.abs(val), 1);
					graphics.setColor(new Color(val < 0 ? abs_val : 0f, val > 0 ? abs_val : 0f, 0f, abs_val));
					graphics.drawLine((int) x_input, (int) (y_min + (node + 1) * y_spacer_input), (int) x_output,
							(int) (y_min + (nextNode + 1) * y_spacer_output));
				}
			}
		}

		// hidden to output
		double y_hidden = (y_max - y_min) / (double) (hiddennodes[hiddennodes.length - 1] + 1);
		double x_hidden = (x_max - x_min) / (double) (hiddennodes.length + 1);

		for (int i = 0; i < hiddennodes[hiddennodes.length - 1]; i++) {
			double y_output = (y_max - y_min) / (double) (outputnodes + 1);

			for (int j = 0; j < outputnodes; j++) {
				float val = (float) this.who.data[j][i];
				float abs_val = Math.min(Math.abs(val), 1);
				graphics.setColor(new Color(val < 0 ? abs_val : 0f, val > 0 ? abs_val : 0f, 0f, abs_val));
				graphics.drawLine((int) (x_max - x_hidden), (int) (y_min + (i + 1) * y_hidden), (x_max),
						(int) (y_min + (j + 1) * y_output));
			}
		}

		// draw nodes
		graphics.setColor(Color.RED);

		// intput nodes
		double y = (y_max - y_min) / (double) (inputnodes + 1);
		for (int i = 0; i < inputnodes; i++)
			graphics.fillOval(x_min - radius, (int) (y_min + (i + 1) * y - radius), diameter, diameter);

		// hidden nodes
		double x = (x_max - x_min) / (double) (hiddennodes.length + 1);
		for (int layer = 0; layer < hiddennodes.length; layer++) {
			y = (y_max - y_min) / (double) (hiddennodes[layer] + 1);
			for (int i = 0; i < hiddennodes[layer]; i++)
				graphics.fillOval((int) (x_min + x * (layer + 1) - radius), (int) (y_min + (i + 1) * y - radius),
						diameter, diameter);
		}

		// output nodes
		y = (y_max - y_min) / (double) (outputnodes + 1);
		for (int i = 0; i < outputnodes; i++)
			graphics.fillOval(x_max - radius, (int) (y_min + (i + 1) * y - radius), diameter, diameter);

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
	 * @return Gibt eine Kopie des neuronalen Netzes zur&uumlck, welche in die exakt
	 *         gleichen Werte hat, wie das originale Netz, aber keinen Bezug hat.
	 * @since 1.0.0
	 */
	public NeuralNetwork copy() {
		NeuralNetwork output = null;
		output = new NeuralNetwork(this.learningrate, this.inputnodes, this.outputnodes, this.hiddennodes);
		output.who = this.who.copy();
		output.wih = this.wih.copy();

		return output;
	}

}