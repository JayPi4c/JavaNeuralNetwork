package com.JayPi4c;

/**
 * 
 * @author JayPi4c
 * @version 1.0.0
 * @since 1.1.0
 */
public class GenericNeuralNetwork extends NeuralNetwork {

	private static final long serialVersionUID = 8008374505415220211L;

	private double mutationRate;

	/**
	 * 
	 * @param inputnodes
	 * @param hiddennodes
	 * @param outputnodes
	 * @param learningrate
	 * @param mutationRate
	 * @since 1.1.0
	 */
	public GenericNeuralNetwork(double learningrate, double mutationRate, int inputnodes, int outputnodes,
			int... hiddennodes) {
		super(learningrate, inputnodes, outputnodes, hiddennodes);
		this.mutationRate = mutationRate;
	}

	/**
	 * Diese Funktion ruft {@link GenericNeuralNetwork#mutate(double)} mit der im
	 * Konstruktor definierten mutationRate auf.
	 * 
	 * @return mutated instance of calling object
	 * @since 1.1.0
	 */
	public GenericNeuralNetwork mutate() {
		return mutate(this.mutationRate);
	}

	/**
	 * Durch eine Mutation des Neuronalen Netzes werden ein Anteil (mutationrate)
	 * der Gewichte im Netz zuf&aumlllig neu festgelegt.
	 * 
	 * Durch diese Funktion wird das aufrufende Neuronale Netz Objekt bearbeitet,
	 * das heisst, dass der R&uumlckgabewert dieser Funktion nicht zwangsl&aumlufig
	 * benutzt werden muss.
	 * 
	 * Diese Funktion soll eigentlich in eine weitere, spezifischere Klasse
	 * untergebracht werden. (GenericNeuralNetwork)
	 * 
	 * @param mutationrate Der Anteil der Gewichte, die neu festgelegt werden.
	 * @return Gibt das mutierte Neuronale Netz zur&uumlck.
	 * @since 1.0.0
	 */
	public GenericNeuralNetwork mutate(double mutationRate) {
		double weights[][] = this.wih.toArray();
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				if (Math.random() < mutationRate)
					weights[i][j] = Math.random() - 0.5;
			}
		}
		wih = new Matrix(weights);
		// TODO whh
		weights = who.toArray();
		for (int i = 0; i < weights.length; i++) {
			for (int j = 0; j < weights[0].length; j++) {
				if (Math.random() < mutationRate)
					weights[i][j] = Math.random() - 0.5;
			}
		}
		who = new Matrix(weights);
		return this;
	}

	// TODO: crossover
	/**
	 * Durch einen Crossover zweier Neuraler Netze wird, wie bei in echter DNA, eine
	 * zuf&aumlllige Mischung dieser beiden Netze erstellt.
	 * 
	 * @deprecated Diese Funktion soll eigentlich in eine weitere, spezifischere
	 *             Klasse untergebracht werden. (GenericNeuralNetwork)
	 * @param other Das andere Neuronale Netz, welches mit dem Neuronalen Netz
	 *              gemischt werden soll.
	 * @param rate  Die Mutationsrate.
	 * @return
	 */
	@Deprecated
	public NeuralNetwork crossover(NeuralNetwork other, double rate) {
		return null;
	}

	// TODO: subclass: GenericNeuralNetwork with crossover and mutation

	@Override
	public GenericNeuralNetwork copy() {
		GenericNeuralNetwork output = new GenericNeuralNetwork(this.learningrate, this.mutationRate, this.inputnodes,
				this.outputnodes, this.hiddennodes);
		for (int i = 0; i < this.whh.length; i++)
			output.whh[i] = whh[i].copy();

		output.who = this.who.copy();
		output.wih = this.wih.copy();

		return output;
	}

}