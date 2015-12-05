package jml.classification;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import jml.matlab.Matlab;
import static jml.matlab.Matlab.sum;
import static jml.matlab.Matlab.zeros;
import static jml.matlab.Matlab.repmat;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.eps;
import static jml.matlab.Matlab.log;
import static jml.matlab.Matlab.sigmoid;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.rdivide;
import jml.options.Options;
import jml.optimization.NonnegativePLBFGS;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;


/**
 * Multi-class logistic regression by using projected limited-memory
 * BFGS method. Projection matrix is constrained to be non-negative.
 * <p/>
 * We aim to minimize the cross-entropy error function defined by
 * <p/>
 * E(W) = -ln{p(T|w1, w2,..., wK)} / N = -sum_n{sum_k{t_{nk}ln(v_nk)}} / N,
 * <p/>where \nabla E(W) = X * (V - T) / N and v_nk = P(C_k|x_n).
 * 
 * @version 1.0 Jan. 12th, 2013
 * 
 * @author Mingjie Qian
 */
public class LogisticRegressionMCNonnegativePLBFGS extends Classifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2215716367007018997L;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = { {3.5, 4.4, 1.3},
						    {5.3, 2.2, 0.5},
						    {0.2, 0.3, 4.1},
						    {-1.2, 0.4, 3.2},
						    {1, 1, 1} };
		
		double[][] labels = { {1, 0, 0},
				  			  {0, 1, 0},
				  			  {0, 0, 1} };
		
		RealMatrix Y = new BlockRealMatrix(labels);
		
		Options options = new Options();
		options.epsilon = 1e-5;
		
		Classifier logReg = new LogisticRegressionMCNonnegativePLBFGS(options);
		logReg.feedData(data);
		logReg.feedLabels(labels);
		long start = System.currentTimeMillis();
		logReg.train();
		
		System.out.println("Projection matrix:");
		Matlab.printMatrix(logReg.getProjectionMatrix());
		
		System.out.println("Ground truth:");
		Matlab.printMatrix(Y);
		RealMatrix Y_pred = logReg.predictLabelScoreMatrix(data);
		System.out.println("Predicted probability matrix:");
		Matlab.printMatrix(Y_pred);
		Y_pred = logReg.predictLabelMatrix(data);
		System.out.println("Predicted label matrix:");
		Matlab.printMatrix(Y_pred);
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		System.out.format("Elapsed time: %.3f seconds\n", elapsedTime);
		
	}

	public LogisticRegressionMCNonnegativePLBFGS(Options options) {
		super(options);
	}

	@Override
	public void train() {
		
		int nFea = this.nFeature;
		int K = this.nClass;
		
		RealMatrix A = null;
		RealMatrix V = null;
		RealMatrix G = null;
		
		double fval = 0;
		
		/* Minimize the cross-entropy error function defined by
		 * E (W) = −ln p (T|w1,w2, · · · ,wK) / nSample
		 * Gradient: G = X * (V - Y) / nSample
		 */
		W = repmat(zeros(nFea, 1), new int[]{1, K});
		A = X.transpose().multiply(W);
		V = sigmoid(A);
		G = X.multiply(V.subtract(Y)).scalarMultiply(1.0 / nExample);
		fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nExample;
		
		boolean flags[] = null;
		while (true) {
			flags = NonnegativePLBFGS.run(G, fval, epsilon, W);
			if (flags[0])
				break;
			A = X.transpose().multiply(W);
			V = sigmoid(A);
			fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nExample;
			if (flags[1]) {
				G = rdivide(X.multiply(V.subtract(Y)), nExample);
				/*fprintf("G:\n");
				disp(G);*/
			}
		}
		
	}

	@Override
	public RealMatrix predictLabelScoreMatrix(RealMatrix X) {
		RealMatrix A = X.transpose().multiply(W);
		return Matlab.sigmoid(A);
	}

	@Override
	public void loadModel(String filePath) {
		
		System.out.println("Loading model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			W = (RealMatrix)ois.readObject();
			IDLabelMap = (int[])ois.readObject();
			nClass = IDLabelMap.length;
			ois.close();
			System.out.println("Model loaded.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
	}

	@Override
	public void saveModel(String filePath){

		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(W);
			oos.writeObject(IDLabelMap);
			oos.close();
			System.out.println("Model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}


