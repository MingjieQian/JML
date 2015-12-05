package jml.classification;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

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
import jml.optimization.LBFGS;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;


/**
 * Multi-class logistic regression by using limited-memory BFGS method.
 * <p/>
 * We aim to minimize the cross-entropy error function defined by
 * <p/>
 * E(W) = -ln{p(T|w1, w2,..., wK)} / N = -sum_n{sum_k{t_{nk}ln(v_nk)}} / N,
 * <p/>where \nabla E(W) = X * (V - T) / N and v_nk = P(C_k|x_n).
 * 
 * @version 1.0 Jan. 10th, 2013
 * 
 * @author Mingjie Qian
 */
public class LogisticRegressionMCLBFGS extends Classifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 337850590370432119L;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = { {3.5, 4.4, 1.3},
						    {5.3, 2.2, 0.5},
						    {0.2, 0.3, 4.1},
						    {-1.2, 0.4, 3.2},
						    {1, 1, 1}
						    };
		
		double[][] labels = { {1, 0, 0},
				  			  {0, 1, 0},
				  			  {0, 0, 1}
				  			  };
		
		RealMatrix Y = new BlockRealMatrix(labels);
		
		Options options = new Options();
		options.epsilon = 1e-5;
		
		Classifier logReg = new LogisticRegressionMCLBFGS(options);
		logReg.feedData(data);
		logReg.feedLabels(labels);
		long start = System.currentTimeMillis();
		logReg.train();
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		System.out.format("Elapsed time: %.3f seconds\n", elapsedTime);
		
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
		
		logReg.saveModel("LogReg-Model.dat");
		((LogisticRegressionMCLBFGS)logReg).saveModel2("LogReg-Model2.dat");
		
		Classifier logReg2 = new LogisticRegressionMCLBFGS();
		logReg2.loadModel("LogReg-Model.dat");
		
		Y_pred = logReg2.predictLabelScoreMatrix(data);
		System.out.println("Predicted probability matrix:");
		Matlab.printMatrix(Y_pred);
		
		// Y_pred = logReg2.predict(data);
		// System.out.println("Predicted probability matrix:");
		Matlab.display(logReg2.predict(data));
		
	}
	
	/**
     * Run this module with arguments in a {@code String} array.
     * 
     * @param args command line arguments
     * 
     */
	public static void run(String[] args) {
		
	}
	
	Model model;
	
	public LogisticRegressionMCLBFGS() {
		super();
		model = null;
	}
	
	public LogisticRegressionMCLBFGS(double epsilon) {
		super();
		this.epsilon = epsilon;
		model = null;
	}
	
	public LogisticRegressionMCLBFGS(Options options) {
		super(options);
		model = null;
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
			flags = LBFGS.run(G, fval, epsilon, W);
			if (flags[0])
				break;
			A = X.transpose().multiply(W);
			V = sigmoid(A);
			fval = -sum(sum(times(Y, log(plus(V, eps))))).getEntry(0, 0) / nExample;
			if (flags[1])
				G = rdivide(X.multiply(V.subtract(Y)), nExample);
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
	
	public void saveModel2(String filePath){

		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(new Model(W));
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

class Model implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -502782622576249525L;
	public RealMatrix W;
	public Model(RealMatrix W) {
		this.W = W;
	}
	
}


