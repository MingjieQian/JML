package jml.classification;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import jml.matlab.Matlab;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.sum;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.eps;
import static jml.matlab.Matlab.log;
import static jml.matlab.Matlab.sigmoid;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.uminus;
import static jml.matlab.Matlab.zeros;
import jml.options.Options;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;


/**
 * Multi-class logistic regression by using gradient descent method.
 * <p/>
 * We aim to minimize the cross-entropy error function defined by
 * <p/>
 * E(W) = -ln{p(T|w1, w2,..., wK)} / N = -sum_n{sum_k{t_{nk}ln(v_nk)}} / N,
 * <p/>where \nabla E(W) = X * (V - T) / N and v_nk = P(C_k|x_n).
 * 
 * @version 1.0 April 3th, 2012
 * 
 * @author Mingjie Qian
 */
public class LogisticRegressionMCGradientDescent extends Classifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 795431672777823665L;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = { {3.5, 4.4, 1.3},
						    {5.3, 2.2, 0.5},
						    {0.2, 0.3, 4.1},
						    {-1.2, 0.4, 3.2} };
		
		double[][] labels = { {1, 0, 0},
				  			  {0, 1, 0},
				  			  {0, 0, 1} };
		
		RealMatrix Y = new BlockRealMatrix(labels);
		
		Options options = new Options();
		options.epsilon = 1e-5;
		
		Classifier logReg = new LogisticRegressionMCGradientDescent(options);
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
	
	public LogisticRegressionMCGradientDescent(Options options) {
		super(options);
	}

	@Override
	public void train() {
		
		double alpha = 0.4;
		double beta = 0.8;
		int K = this.nClass;
		W = Matlab.repmat(zeros(nFeature, 1), new int[]{1, K});
		ArrayList<Double> J = new ArrayList<Double>();
		J.add(-Matlab.sum(
				Matlab.sum(
					Matlab.times(
						Y, Matlab.log(Matlab.ones(Matlab.size(Y)).scalarMultiply(1d / K))
						)
					)
				).getEntry(0, 0) / nExample);
		
		// J.add(Double.POSITIVE_INFINITY);
		/*RealMatrix I = Matlab.eye(K);
		RealMatrix H = Matlab.zeros(K * p, K * p);*/
		
		int cnt = 0;
		// RealMatrix W_old = W.copy();
		// RealMatrix W_old = null;
		RealMatrix A = null;
		RealMatrix V = null;
		RealMatrix G = null;
		RealMatrix p = null;
		
		double t = 10;
	    RealMatrix W_t = null;
	    RealMatrix A_t = null;
	    RealMatrix V_t = null;
	    
		double d = 0;
		double z = 0;
		// double dJ = 0;
		double fval = 0;
		double fval_t = 0;
		while (true) {

			// W_old.setSubMatrix(W.getData(), 0, 0);
			// W_old = W;
			A = X.transpose().multiply(W);
			V = Matlab.sigmoid(A);
			G = X.multiply(V.subtract(Y)).scalarMultiply(1.0 / nExample);
			
			// W.setSubMatrix(W_old.subtract(G.scalarMultiply(K * 5)).getData(), 0, 0);
			// W.setSubMatrix(Matlab.reshape(Matlab.vec(W_old).subtract(Matlab.vec(G).scalarMultiply(K * 5)), new int[]{nFeature, nClass}).getData(), 0, 0);
			// W = W_old.subtract(G.scalarMultiply(K * 2.5));
			
			cnt = cnt + 1;
			fval = -sum(sum(times(Y, log(V.scalarAdd(Double.MIN_VALUE))))).getEntry(0, 0) / nExample;
			J.add(fval);

			// Backtracking line search
			// p is a decreasing step
			p = uminus(G);
			// t is the step length, for gradient descent method,
			// a large step length could used for Armijo condition
		    t = 10.0;
		    z = innerProduct(G, G);
		    while (true) {
		        W_t = plus(W, times(t, p));
		        A_t = X.transpose().multiply(W_t);
		        V_t = sigmoid(A_t);
		        fval_t = -sum(sum(times(Y, log(plus(V_t, eps))))).getEntry(0, 0) / nExample;
		        if (fval_t <= fval - alpha * t * z)
		            break;
		        else
		            t = beta * t;
		    }
		    
		    // System.out.println(t);
		    W = W_t;
		        
			/*dJ = Math.abs(J.get(J.size() - 1) - J.get(J.size() - 2));
			System.out.format("Iteration %d, delta J: %f, J: %f\n", cnt, dJ, J.get(J.size() - 1));*/

			// d = Matlab.norm(W.subtract(W_old), "fro");
		    d = norm(G);
			System.out.format("Iteration %d, norm of grad: %f, J: %f\n", cnt, d, J.get(J.size() - 1));

			if  (d < epsilon) {
				break;
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
