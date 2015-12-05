package jml.classification;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

import jml.matlab.Matlab;
import static jml.matlab.Matlab.sum;
import static jml.matlab.Matlab.zeros;
import static jml.matlab.Matlab.repmat;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.eps;
import static jml.matlab.Matlab.log;
import static jml.matlab.Matlab.sigmoid;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.innerProduct;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.uminus;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.minus;

import jml.options.Options;

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
public class LogisticRegressionMCLBFGS_Ori extends Classifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2027322010064300037L;

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
		
		Classifier logReg = new LogisticRegressionMCLBFGS_Ori(options);
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

	public LogisticRegressionMCLBFGS_Ori(Options options) {
		super(options);
	}

	@Override
	public void train() {
		
		double alpha = 0.2;
		double beta = 0.5;
		double H0 = 1;
		int m = 10;
		
		double H = 0;
		RealMatrix A = null;
		RealMatrix V = null;
		RealMatrix G = null;
		
		double t = 10;
	    RealMatrix W_t = null;
	    RealMatrix A_t = null;
	    RealMatrix V_t = null;
	    
	    RealMatrix W_pre = null;
	    RealMatrix G_pre = null;
	    
		double fval = 0;
		double fval_t = 0;
		int nFea = this.nFeature;
		int K = this.nClass;
		W = repmat(zeros(nFea, 1), new int[]{1, K});
		A = X.transpose().multiply(W);
		V = sigmoid(A);
		G = X.multiply(V.subtract(Y)).scalarMultiply(1.0 / nExample);
		ArrayList<Double> J = new ArrayList<Double>();
		
		fval = -sum(sum(times(Y, log(V.scalarAdd(Double.MIN_VALUE))))).getEntry(0, 0) / nExample;
		J.add(fval);
		System.out.format("Initial ofv: %g\n", fval);
		
		RealMatrix s_k = null;
		RealMatrix y_k = null;
		double rou_k = 0;
		
		RealMatrix s_k_i = null;
		RealMatrix y_k_i = null;
		Double rou_k_i = null;
		
		LinkedList<RealMatrix> s_ks = new LinkedList<RealMatrix>();
		LinkedList<RealMatrix> y_ks = new LinkedList<RealMatrix>();
		LinkedList<Double> rou_ks = new LinkedList<Double>();	
		
		Iterator<RealMatrix> iter_s_ks = null;
		Iterator<RealMatrix> iter_y_ks = null;
		Iterator<Double> iter_rou_ks = null;
		
		double[] a = new double[m];
		double b = 0;
		double z = 0;
		
		RealMatrix p = null;
		RealMatrix q = null;
		RealMatrix r = null;
				
		int k = 0;
		while (true) {
			
			if (norm(G) < epsilon) {
				System.out.println("Converge.");
				break;
			}
			
			if (k == 0) {
				H = H0;
			} else {
				H = innerProduct(s_k, y_k) / innerProduct(y_k, y_k);
			}
			
			// Algorithm 7.4
			q = G;
			iter_s_ks = s_ks.descendingIterator();
			iter_y_ks = y_ks.descendingIterator();
			iter_rou_ks = rou_ks.descendingIterator();
			for (int i = s_ks.size() - 1; i >= 0; i--) {
				s_k_i = iter_s_ks.next();
				y_k_i = iter_y_ks.next();
				rou_k_i = iter_rou_ks.next();
				a[i] = rou_k_i * innerProduct(s_k_i, q);
				q = q.subtract(times(a[i], y_k_i));
			}
			r = times(H, q);
			iter_s_ks = s_ks.iterator();
			iter_y_ks = y_ks.iterator();
			iter_rou_ks = rou_ks.iterator();
			for (int i = 0; i < s_ks.size(); i++) {
				s_k_i = iter_s_ks.next();
				y_k_i = iter_y_ks.next();
				rou_k_i = iter_rou_ks.next();
				b = rou_k_i * innerProduct(y_k_i, r);
				r = r.add(times(a[i] - b, s_k_i));
			}
			// p is a decreasing direction
			p = uminus(r);
			
			t = 1;
			// z is always less than 0
			z = innerProduct(G, p);
			while (true) {
				W_t = plus(W, times(t, p));
				A_t = X.transpose().multiply(W_t);
				V_t = sigmoid(A_t);
				fval_t = -sum(sum(times(Y, log(plus(V_t, eps))))).getEntry(0, 0) / nExample;
				if (fval_t <= fval + alpha * t * z) {
					break;
				} else {
					t = beta * t;
				}
			}
			
			W_pre = W;    
		    G_pre = G;
		    
		    if (Math.abs(fval_t - fval) < eps)
		    	break;
	        
		    fval = fval_t;
		    J.add(fval);
		    System.out.format("Iter %d, ofv: %g\n", k + 1, fval);
		    
		    // Template {
		    W = W_t;
		    V = V_t;
		    G = rdivide(X.multiply(V.subtract(Y)), nExample);
			// }
		    
		    s_k = W.subtract(W_pre);
		    y_k = minus(G, G_pre);
		    rou_k = 1 / innerProduct(y_k, s_k);
		    
		    // Now s_ks, y_ks, and rou_ks all have k elements
		    if (k >= m) {
		    	s_ks.removeFirst();
		    	y_ks.removeFirst();
		    	rou_ks.removeFirst();
		    }
		    s_ks.add(s_k);
	    	y_ks.add(y_k);
	    	rou_ks.add(rou_k);
		    
		    k = k + 1;
			
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

