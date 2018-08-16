package com.JayPi4c;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

import com.JayPi4c.Matrix.Matrix;
import com.JayPi4c.NeuralNetwork.NeuralNetwork;

public class Test {
	public static void main(String args[]) throws Exception {

		makeSerializedTrainedNetwork();
		String absolutePath = new File(".").getAbsolutePath();
		File file = new File(absolutePath);
		absolutePath = file.getParentFile().toString();
		NeuralNetwork nn = NeuralNetwork.deserialize(new File(absolutePath + "/NeuralNetwork.nn"));

		double input[][] = { { 0.9 } };
		double target[][] = { { 0.81 } };
		System.out.println("Prediction: ");
		nn.query(input).print();
		System.out.println("Target: ");
		new Matrix(target).print();
	}

	public static void makeSerializedTrainedNetwork() throws IllegalAccessException, IllegalArgumentException,
			InvocationTargetException, NoSuchMethodException, SecurityException, IOException {

		NeuralNetwork nn = new NeuralNetwork(1, 5, 1, 0.1);
		double inputs[][] = { { 1 } };
		double targets[][] = { { 1 } };
		System.out.println("Prediction: ");
		nn.query(new Matrix(inputs)).print();
		System.out.println("Target: ");
		new Matrix(targets).print();

		/*
		 * // Diese Schleife kann das !Xor trainieren for (int i = 0; i < 10000000; i++)
		 * { double rand = Math.random(); if (rand < 0.25) { double input[][] = { { 1, 1
		 * } }; double target[][] = { { 1 } }; nn.train(new Matrix(input), new
		 * Matrix(target)); } else if (rand < 0.5) { double input[][] = { { 1, 0 } };
		 * double target[][] = { { 0 } }; nn.train(new Matrix(input), new
		 * Matrix(target)); } else if (rand < 0.75) { double input[][] = { { 0, 1 } };
		 * double target[][] = { { 0 } }; nn.train(new Matrix(input), new
		 * Matrix(target)); } else { double input[][] = { { 0, 0 } }; double target[][]
		 * = { { 1 } }; nn.train(new Matrix(input), new Matrix(target)); } if (i %
		 * 100000 == 0) System.out.println((double) i / (double) 100000 +
		 * " * 100000 geschafft"); }
		 */

		for (int i = 0; i < 1000000; i++) {
			double input[][] = { { Math.random() } };
			double target[][] = { { input[0][0] * input[0][0] } };
			nn.train(new Matrix(input), new Matrix(target));
		}
		System.out.println("Training done!");

		nn.serialize();
	}

}
