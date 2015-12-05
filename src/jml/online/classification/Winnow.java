package jml.online.classification;

import static jml.matlab.Matlab.disp;
import static jml.matlab.Matlab.exp;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.ones;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.size;
import static jml.matlab.Matlab.times;
import static jml.utils.Time.tic;
import static jml.utils.Time.toc;

import org.apache.commons.math.linear.RealMatrix;

/***
 * Java implementation of the online binary classification
 * algorithm Winnow.
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 29th, 2013
 */
public class Winnow extends OnlineBinaryClassifier {
	
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

		double eta = 1;
		OnlineBinaryClassifier winnow = new Winnow(eta);
		int n = data.length;
		for (int i = 0; i < n; i++) {
			winnow.train(data[i], labels[i]);
		}

		System.out.println("Projection matrix:");
		disp(winnow.W);

		// Prediction
		fprintf("Ground truth:\n");
		for (int i = 0; i < n; i++) {
			fprintf("Y_true for sample %d: %d\n", i + 1, labels[i]);
		}
		fprintf("Prediction:\n");
		for (int i = 0; i < n; i++) {
			int yt = winnow.predict(data[i]);
			fprintf("Y_pred for sample %d: %d\n", i + 1, yt);
		}

		System.out.format("Elapsed time: %.1f seconds.\n", toc());

	}

	public double eta;

	public Winnow(double eta) {
		super();
		this.eta = eta;
	}

	@Override
	public void train(RealMatrix X, int y) {

		if (W == null) {
			W = rdivide(ones(size(X, 1), 1), size(X, 1));
		}

		if ((2 * innerProduct(W, X) - 1) * y <= 0) {
			W = times(W, exp(times(X, -2 * eta * y)));
		}

	}

	@Override
	public int predict(RealMatrix Xt) {
		return 2 * innerProduct(W, Xt) - 1 >= 0 ? 1 : -1;
	}

}
