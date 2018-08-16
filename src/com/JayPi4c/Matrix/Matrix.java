package com.JayPi4c.Matrix;

import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.concurrent.ThreadLocalRandom;

public class Matrix implements Serializable {

	private static final long serialVersionUID = -1611903368454112326L;
	int rows, cols;
	double data[][];

	public Matrix() {
	}

	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		this.data = new double[this.rows][this.cols];
	}

	public Matrix(double data[][]) {
		this.rows = data.length;
		this.cols = data[0].length;
		this.data = new double[this.rows][this.cols];
		for (int row = 0; row < this.rows; row++)
			for (int col = 0; col < this.cols; col++)
				this.data[row][col] = data[row][col];
	}

	public double[][] toArray() {
		double output[][] = new double[this.rows][this.cols];
		for (int row = 0; row < this.rows; row++)
			for (int col = 0; col < this.cols; col++)
				output[row][col] = this.data[row][col];
		return output;
	}

	public static double[][] toArray(Matrix m) {
		double output[][] = new double[m.rows][m.cols];
		for (int row = 0; row < m.rows; row++)
			for (int col = 0; col < m.cols; col++)
				output[row][col] = m.data[row][col];
		return output;
	}

	public Matrix copy() {
		return new Matrix(this.data);
	}

	public Matrix fill(double d) {
		for (int row = 0; row < this.rows; row++) {
			for (int col = 0; col < this.cols; col++) {
				this.data[row][col] = d;
			}
		}
		return this;
	}

	/**
	 * Verwendung:<br>
	 * Matrix m = new Matrix(3, 3);<br>
	 * m.fill(new Matrix(), Matrix.class.getMethod("random"));
	 * 
	 * @param obj
	 * @param m   Muss eine Funktion sein, die einen double zur&uumlck gibt und
	 *            keine Parameter annimmt
	 * @return
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 * @throws InvocationTargetException
	 */
	public Matrix fill(Object obj, Method m)
			throws IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		for (int row = 0; row < this.rows; row++) {
			for (int col = 0; col < this.cols; col++) {
				this.data[row][col] = (double) m.invoke(obj);
			}
		}
		return this;
	}

	/**
	 * 
	 * Verwendung:<br>
	 * Matrix m = new Matrix(3, 3);<br>
	 * m.map(new Matrix(), Matrix.class.getMethod("sigmoid", double.class));
	 * 
	 * @param obj
	 * @param m
	 * @return
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 * @throws InvocationTargetException
	 */
	public Matrix map(Object obj, Method m)
			throws IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		for (int row = 0; row < this.rows; row++) {
			for (int col = 0; col < this.cols; col++) {
				this.data[row][col] = (double) m.invoke(obj, this.data[row][col]);
			}
		}
		return this;
	}

	public Matrix randomize(double min, double max) {
		for (int row = 0; row < this.rows; row++) {
			for (int col = 0; col < this.cols; col++) {
				this.data[row][col] = ThreadLocalRandom.current().nextDouble(min, max + 0.000001);
			}
		}
		return this;
	}

	public Matrix randomize() {
		for (int row = 0; row < this.rows; row++) {
			for (int col = 0; col < this.cols; col++) {
				this.data[row][col] = ThreadLocalRandom.current().nextDouble();
			}
		}
		return this;
	}

	public void print() {
		System.out.println("-------------------------------------------------");
		for (int row = 0; row < this.rows; row++) {
			for (int col = 0; col < this.cols; col++) {
				System.out.print(this.data[row][col] + "\t");
			}
			System.out.println();
		}
		System.out.println("-------------------------------------------------");
	}

	public double random() {
		return Math.random();
	}

	public Matrix transpose() {
		Matrix m = new Matrix(this.cols, this.rows);
		for (int row = 0; row < this.rows; row++)
			for (int col = 0; col < this.cols; col++)
				m.data[col][row] = this.data[row][col];
		this.data = m.data;
		this.rows = m.rows;
		this.cols = m.cols;

		return this;
	}

	public static Matrix transpose(Matrix matrix) {
		Matrix newMatrix = new Matrix(matrix.cols, matrix.rows);
		for (int row = 0; row < matrix.rows; row++)
			for (int col = 0; col < matrix.cols; col++)
				newMatrix.data[col][row] = matrix.data[row][col];
		return newMatrix;
	}

	public Matrix dot(Matrix m) {
		if (this.cols != m.rows) {
			System.out.println("A's cols and B's rows must match!");
			return null;
		}
		Matrix newMatrix = new Matrix(this.rows, m.cols);
		for (int row = 0; row < newMatrix.rows; row++) {
			for (int col = 0; col < newMatrix.cols; col++) {
				double sum = 0;
				for (int j = 0; j < this.cols; j++) {
					sum += this.data[row][j] * m.data[j][col];
				}
				newMatrix.data[row][col] = sum;
			}
		}
		this.data = newMatrix.data;
		this.rows = newMatrix.rows;
		this.cols = newMatrix.cols;

		return this;
	}

	public static Matrix dot(Matrix a, Matrix b) {
		if (a.cols != b.rows) {
			System.out.println("A's cols and B's rows must match!");
			return null;
		}
		Matrix newMatrix = new Matrix(a.rows, b.cols);
		for (int row = 0; row < newMatrix.rows; row++) {
			for (int col = 0; col < newMatrix.cols; col++) {
				double sum = 0;
				for (int j = 0; j < a.cols; j++) {
					sum += a.data[row][j] * b.data[j][col];
				}
				newMatrix.data[row][col] = sum;
			}
		}
		return newMatrix;
	}

	public Matrix sub(Matrix m) {
		if (m.cols != this.cols || m.rows != this.rows) {
			System.out.println("rows and columns must match!");
			return null;
		}
		for (int row = 0; row < this.rows; row++)
			for (int col = 0; col < this.cols; col++)
				this.data[row][col] -= m.data[row][col];
		return this;
	}

	public static Matrix sub(Matrix a, Matrix b) {
		if (a.cols != b.cols || a.rows != b.rows) {
			System.out.println("rows and columns must match!");
			return null;
		}
		Matrix newMatrix = new Matrix(a.rows, a.cols);
		for (int row = 0; row < a.rows; row++)
			for (int col = 0; col < a.cols; col++)
				newMatrix.data[row][col] = a.data[row][col] - b.data[row][col];
		return newMatrix;
	}

	public static Matrix sub(double d, Matrix m) {
		Matrix newMatrix = new Matrix(m.rows, m.cols);
		for (int row = 0; row < newMatrix.rows; row++)
			for (int col = 0; col < newMatrix.cols; col++)
				newMatrix.data[row][col] = d - m.data[row][col];
		return newMatrix;
	}

	public Matrix add(Matrix m) {
		if (this.cols != m.cols || this.rows != m.rows) {
			System.out.println("rows and columns must match!");
			return null;
		}
		for (int row = 0; row < this.rows; row++)
			for (int col = 0; col < this.cols; col++)
				this.data[row][col] += m.data[row][col];
		return this;
	}

	public static Matrix add(Matrix a, Matrix b) {
		if (a.cols != b.cols || a.rows != b.rows) {
			System.out.println("rows and columns must match!");
			return null;
		}
		Matrix newMatrix = new Matrix(a.rows, a.cols);
		for (int row = 0; row < a.rows; row++)
			for (int col = 0; col < a.cols; col++)
				newMatrix.data[row][col] = a.data[row][col] + b.data[row][col];
		return newMatrix;
	}

	public Matrix mult(double scl) {
		for (int row = 0; row < this.rows; row++)
			for (int col = 0; col < this.cols; col++)
				this.data[row][col] *= scl;
		return this;
	}

	public Matrix mult(Matrix m) {
		if (this.cols != m.cols || this.rows != m.rows) {
			System.out.println("rows and columns must match!");
			return null;
		}
		for (int row = 0; row < this.rows; row++)
			for (int col = 0; col < this.cols; col++)
				this.data[row][col] *= m.data[row][col];
		return this;
	}

	public static Matrix mult(Matrix m, double scl) {
		Matrix newMatrix = new Matrix(m.rows, m.cols);
		for (int row = 0; row < newMatrix.rows; row++)
			for (int col = 0; col < newMatrix.cols; col++)
				newMatrix.data[row][col] = scl * m.data[row][col];
		return newMatrix;
	}

	public static Matrix mult(Matrix a, Matrix b) {
		if (a.cols != b.cols || a.rows != b.rows) {
			System.out.println("rows and columns must match!");
			return null;
		}
		Matrix newMatrix = new Matrix(a.rows, b.cols);
		for (int row = 0; row < newMatrix.rows; row++)
			for (int col = 0; col < newMatrix.cols; col++)
				newMatrix.data[row][col] = a.data[row][col] * b.data[row][col];
		return newMatrix;
	}

}