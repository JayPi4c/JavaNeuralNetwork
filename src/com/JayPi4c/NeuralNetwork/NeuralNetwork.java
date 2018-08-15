package com.JayPi4c.NeuralNetwork;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;

import com.JayPi4c.Matrix.Matrix;

public class NeuralNetwork implements Serializable {

	private static final long serialVersionUID = 5795625580326323029L;

	private int inputnodes, hiddennodes, outputnodes;
	private double learningrate;

	private Matrix wih, who;

	public NeuralNetwork() {
	}

	public NeuralNetwork(int inputnodes, int hiddennodes, int outputnodes, double learningrate)
			throws IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException,
			SecurityException {
		this.inputnodes = inputnodes;
		this.hiddennodes = hiddennodes;
		this.outputnodes = outputnodes;
		this.learningrate = learningrate;
		this.wih = new Matrix(this.hiddennodes, this.inputnodes).map(new NeuralNetwork(),
				NeuralNetwork.class.getMethod("random", double.class));
		this.who = new Matrix(this.outputnodes, this.hiddennodes).map(new NeuralNetwork(),
				NeuralNetwork.class.getMethod("random", double.class));

	}

	// ****************************************************************************************************************//

	public Matrix query(Matrix inputs_list) throws IllegalAccessException, IllegalArgumentException,
			InvocationTargetException, NoSuchMethodException, SecurityException {
		Matrix inputs = Matrix.transpose(inputs_list);
		Matrix hidden_inputs = Matrix.dot(this.wih, inputs);
		Matrix hidden_outputs = hidden_inputs.map(new NeuralNetwork(),
				NeuralNetwork.class.getMethod("sigmoid", double.class));
		Matrix final_inputs = Matrix.dot(this.who, hidden_outputs);
		Matrix final_outputs = final_inputs.map(new NeuralNetwork(),
				NeuralNetwork.class.getMethod("sigmoid", double.class));
		return final_outputs;
	}

	public Matrix query(double inputs_list[][]) throws IllegalAccessException, IllegalArgumentException,
			InvocationTargetException, NoSuchMethodException, SecurityException {
		return this.query(new Matrix(inputs_list));
	}

	// ****************************************************************************************************************//

	public void train(Matrix inputs_list, Matrix targets_list) throws IllegalAccessException, IllegalArgumentException,
			InvocationTargetException, NoSuchMethodException, SecurityException {
		Matrix inputs = Matrix.transpose(inputs_list);
		Matrix targets = Matrix.transpose(targets_list);

		Matrix hidden_inputs = Matrix.dot(this.wih, inputs);
		Matrix hidden_outputs = hidden_inputs.map(new NeuralNetwork(),
				NeuralNetwork.class.getMethod("sigmoid", double.class));
		Matrix final_inputs = Matrix.dot(this.who, hidden_outputs);
		Matrix final_outputs = final_inputs.map(new NeuralNetwork(),
				NeuralNetwork.class.getMethod("sigmoid", double.class));

		Matrix output_errors = Matrix.sub(targets, final_outputs);
		Matrix hidden_errors = Matrix.dot(Matrix.transpose(this.who), output_errors);

		this.who.add(Matrix
				.mult(Matrix.dot(Matrix.mult(output_errors, Matrix.mult(final_outputs, Matrix.sub(1, final_outputs))),
						Matrix.transpose(hidden_outputs)), this.learningrate));

		this.wih.add(Matrix
				.mult(Matrix.dot(Matrix.mult(hidden_errors, Matrix.mult(hidden_outputs, Matrix.sub(1, hidden_outputs))),
						Matrix.transpose(inputs)), this.learningrate));

	}

	public void train(double inputs_list[][], double targets_list[][]) throws IllegalAccessException,
			IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException {
		this.train(new Matrix(inputs_list), new Matrix(targets_list));
	}

	// ****************************************************************************************************************//

	public double random(double d) {
		return Math.random() - 0.5;

	}

	public double sigmoid(double x) {
		return (1 / (1 + Math.pow(Math.E, -x)));
	}

	public void serialize() throws IOException {
		String absolutePath = new File(".").getAbsolutePath();
		File file = new File(absolutePath);
		absolutePath = file.getParentFile().toString();
		FileOutputStream fos = new FileOutputStream(new File(absolutePath + "/NeuralNetwork.nn"));
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		oos.writeObject(this);
		oos.close();
	}

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
	 * 
	 * @param f Ein File Object, welches die Daten des Neuronalen Netzes beinhaltet
	 * @return gibt ein Neuronales Netz zur&uumlck
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static NeuralNetwork deserialize(File f) throws IOException, ClassNotFoundException {
		FileInputStream fis = new FileInputStream(f);
		ObjectInputStream ois = new ObjectInputStream(fis);

		NeuralNetwork output = null;
		output = (NeuralNetwork) ois.readObject();
		ois.close();
		return output;
	}

	/**
	 * 
	 * @return Gibt eine Kopie des neuronalen Netzes zur&uumlck, welche in die exakt
	 *         gleichen Werte hat, wie das originale Netz, aber keinen Bezug hat.
	 */
	public NeuralNetwork copy() {
		NeuralNetwork output = null;
		try {
			output = new NeuralNetwork(this.inputnodes, this.hiddennodes, this.outputnodes, this.learningrate);
			output.who = this.who.copy();
			output.wih = this.wih.copy();
		} catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException | NoSuchMethodException
				| SecurityException e) {
			e.printStackTrace();
		}

		return output;
	}

}