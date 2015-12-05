package jml.online.classification;

import static jml.matlab.Matlab.disp;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.size;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.zeros;
import static jml.utils.Time.tic;
import static jml.utils.Time.toc;

import org.apache.commons.math.linear.RealMatrix;

/***
 * Java implementation of the online binary classification
 * algorithm Perceptron.
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 29th, 2013
 */
public class Perceptron extends OnlineBinaryClassifier {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = { {3.5, 4.4, 1.3},
							{5.3, 2.2, 0.5},
							{0.2, 0.3, 4.1},
							{-1.2, 0.4, 3.2} };

		int[] labels = {1, -1, 1, -1};

		// Get elapsed time in seconds
		tic();
				
		OnlineBinaryClassifier perceptron = new Perceptron();
		int n = data.length;
		for (int i = 0; i < n; i++) {
			perceptron.train(data[i], labels[i]);
		}
		
		System.out.println("Projection matrix:");
		disp(perceptron.W);
		
		// Prediction
		fprintf("Ground truth:\n");
		for (int i = 0; i < n; i++) {
			fprintf("Y_true for sample %d: %d\n", i + 1, labels[i]);
		}
		fprintf("Prediction:\n");
		for (int i = 0; i < n; i++) {
			int yt = perceptron.predict(data[i]);
			fprintf("Y_pred for sample %d: %d\n", i + 1, yt);
		}
		
		System.out.format("Elapsed time: %.1f seconds.\n", toc());

	}

	public Perceptron() {
		super();
	}

	@Override
	public void train(RealMatrix X, int y) {
		
		if (W == null) {
			W = zeros(size(X, 1), 1);
		}

		if (innerProduct(W, X) * y <= 0) {
			W = W.add(times(y, X));
		}
		
	}
	
	@Override
	public int predict(RealMatrix Xt) {
		return innerProduct(W, Xt) >= 0 ? 1 : -1;
	}
	
	@Override
	public int predict(double[] Xt) {
		int d = Xt.length;
		double score = 0;
		for (int i = 0; i < d; i++) {
			score += W.getEntry(i, 0) * Xt[i];
		}
		return score >= 0 ? 1 : -1;
	}

}
