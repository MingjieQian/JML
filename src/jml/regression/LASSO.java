package jml.regression;

import static jml.matlab.Matlab.abs;
import static jml.matlab.Matlab.diag;
import static jml.matlab.Matlab.display;
import static jml.matlab.Matlab.eye;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.gt;
import static jml.matlab.Matlab.horzcat;
import static jml.matlab.Matlab.inf;
import static jml.matlab.Matlab.logicalIndexingAssignment;
import static jml.matlab.Matlab.lt;
import static jml.matlab.Matlab.max;
import static jml.matlab.Matlab.minus;
import static jml.matlab.Matlab.mldivide;
import static jml.matlab.Matlab.mtimes;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.not;
import static jml.matlab.Matlab.or;
import static jml.matlab.Matlab.plus;
import static jml.matlab.Matlab.power;
import static jml.matlab.Matlab.rdivide;
import static jml.matlab.Matlab.repmat;
import static jml.matlab.Matlab.size;
import static jml.matlab.Matlab.subplus;
import static jml.matlab.Matlab.sum;
import static jml.matlab.Matlab.times;
import static jml.matlab.Matlab.uminus;
import static jml.matlab.Matlab.vertcat;
import static jml.matlab.Matlab.zeros;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import jml.options.Options;

import org.apache.commons.math.linear.RealMatrix;

/***
 * A Java implementation of LASSO, which solves the following
 * convex optimization problem:
 * </p>
 * min_W 2\1 || Y - X * W ||_F^2 + lambda * || W ||_1</br>
 * where X is an n-by-p data matrix with each row bing a p
 * dimensional data vector and Y is an n-by-ny dependent
 * variable matrix.
 * 
 * @author Mingjie Qian
 * @version 1.0 Jan. 14th, 2013
 */
public class LASSO extends Regression {
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		double[][] data = {{1, 2, 3, 2},
						   {4, 2, 3, 6},
						   {5, 1, 2, 1}};
		
		double[][] depVars = {{3, 2},
							  {2, 3},
							  {1, 4}};

		Options options = new Options();
		options.maxIter = 600;
		options.lambda = 0.05;
		options.verbose = !true;
		options.calc_OV = !true;
		options.epsilon = 1e-5;
		
		Regression LASSO = new LASSO(options);
		LASSO.feedData(data);
		LASSO.feedDependentVariables(depVars);
		
		long start = System.currentTimeMillis();
		LASSO.train();
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		fprintf("Elapsed time: %.3f seconds\n\n", elapsedTime);
		
		fprintf("Projection matrix:\n");
		display(LASSO.W);
		
		RealMatrix Yt = LASSO.predict(data);
		fprintf("Predicted dependent variables:\n");
		display(Yt);
		
	}

	
	/**
	 * Regularization parameter.
	 */
	private double lambda;
	
	/**
	 * If compute objective function values during
	 * the iterations or not.
	 */
	private boolean calc_OV;
	
	/**
	 * If show computation detail during iterations or not.
	 */
	private boolean verbose;

	public LASSO() {
		super();
		lambda = 1;
		calc_OV = false;
		verbose = false;
	}

	public LASSO(double epsilon) {
		super(epsilon);
		lambda = 1;
		calc_OV = false;
		verbose = false;
	}
	
	public LASSO(int maxIter, double epsilon) {
		super(maxIter, epsilon);
		lambda = 1;
		calc_OV = false;
		verbose = false;
	}
	
	public LASSO(double lambda, int maxIter, double epsilon) {
		super(maxIter, epsilon);
		this.lambda = lambda;
		calc_OV = false;
		verbose = false;
	}
	
	public LASSO(Options options) {
		super(options);
		lambda = options.lambda;
		calc_OV = options.calc_OV;
		verbose = options.verbose;
	}

	@Override
	public void train() {
		W = train(X, Y);
	}

	@Override
	public void loadModel(String filePath) {
		
		// System.out.println("Loading regression model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			W = (RealMatrix)ois.readObject();
			ois.close();
			System.out.println("LASSO model loaded.");
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
	public void saveModel(String filePath) {
		
		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(W);
			oos.close();
			System.out.println("LASSO model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

	
	public static RealMatrix train(RealMatrix X, RealMatrix Y, Options options) {

		int p = size(X, 2);
		int ny = size(Y, 2);
		double epsilon = options.epsilon;
		int maxIter = options.maxIter;
		double lambda = options.lambda;
		boolean calc_OV = options.calc_OV;
		boolean verbose = options.verbose;
		
		/*XNX = [X, -X];
		H_G = XNX' * XNX;
		D = repmat(diag(H_G), [1, n_y]);
		XNXTY = XNX' * Y;
	    A = (X' * X + lambda  * eye(p)) \ (X' * Y);*/
		
		RealMatrix XNX = horzcat(X, uminus(X));
		RealMatrix H_G = XNX.transpose().multiply(XNX);
		RealMatrix D = repmat(diag(H_G), 1, ny);
		RealMatrix XNXTY = XNX.transpose().multiply(Y);
		RealMatrix A = mldivide(
						   plus(X.transpose().multiply(X), times(lambda, eye(p))), 
						   X.transpose().multiply(Y)
						   );
		
		/*AA = [subplus(A); subplus(-A)];
		C = -XNXTY + lambda;
		Grad = C + H_G * AA;
		tol = epsilon * norm(Grad);
		PGrad = zeros(size(Grad));*/
		
		RealMatrix AA = vertcat(subplus(A), subplus(uminus(A)));
		RealMatrix C = plus(uminus(XNXTY), lambda);
		RealMatrix Grad = plus(C, mtimes(H_G, AA));
		double tol = epsilon * norm(Grad);
		RealMatrix PGrad = zeros(size(Grad));
		
		ArrayList<Double> J = new ArrayList<Double>();
		double fval = 0;
		// J(1) = sum(sum((Y - X * A).^2)) / 2 + lambda * sum(sum(abs(A)));
		if (calc_OV) {
			fval = sum(sum(power(minus(Y, mtimes(X, A)), 2))).getEntry(0, 0) / 2 +
				   lambda * sum(sum(abs(A))).getEntry(0, 0);
			J.add(fval);
		}
		
		RealMatrix I_k = null;
		RealMatrix I_k_com = null;
		double d = 0;
		RealMatrix tmp = null;
		int k = 0;
		while (true) {	
			
			/*I_k = Grad < 0 | AA > 0;
		    I_k_com = not(I_k);
		    PGrad(I_k) = Grad(I_k);
		    PGrad(I_k_com) = 0;*/
			
			I_k = or(lt(Grad, 0), gt(AA, 0));
			I_k_com = not(I_k);
			PGrad = Grad.copy();
			logicalIndexingAssignment(PGrad, I_k_com, 0);
			
			d = norm(PGrad, inf);
		    if (d < tol) {
		        if (verbose)
		            System.out.println("Converge successfully!");
		        
		        break;
		    }
		    
		    /*for i = 1:2*p
		            AA(i, :) = max(AA(i, :) - (C(i, :) + H_G(i, :) * AA) ./ (D(i, :)), 0);
		    end
		    A = AA(1:p,:) - AA(p+1:end,:);*/
		    
		    for (int i = 0; i < 2 * p; i++) {
		    	tmp = max(
		    			minus(
	    					AA.getRowMatrix(i), 
	    					rdivide(
    							plus(C.getRowMatrix(i), mtimes(H_G.getRowMatrix(i), AA)),
    						    D.getRowMatrix(i)
    						    )
		    				),
		    		    0.0
		    		    );
		    						    
		    	AA.setRowMatrix(i, tmp);
		    }
		    
		    Grad = plus(C, mtimes(H_G,AA));
		    
		    k = k + 1;
		    if (k > maxIter) {
		    	if (verbose)
		    		System.out.println("Maximal iterations");
		    	break;
		    }
		    
		    if (calc_OV) {
				fval = sum(sum(power(minus(Y, mtimes(XNX, AA)), 2))).getEntry(0, 0) / 2 +
						lambda * sum(sum(abs(AA))).getEntry(0, 0);
				J.add(fval);
			}
		    
		    if (k % 10 == 0 && verbose) {
		    	if (calc_OV)
		    		System.out.format("Iter %d - ||PGrad||: %f, ofv: %f\n", k, d, J.get(J.size() - 1));
		    	else
		    		System.out.format("Iter %d - ||PGrad||: %f\n", k, d);

		    }
			
		}
		
		A = minus(
	    		  AA.getSubMatrix(0, p - 1, 0, ny - 1),
	    		  AA.getSubMatrix(p, 2 * p - 1, 0, ny - 1)
	    		  );
		return A;
		
	}
	
	@Override
	public RealMatrix train(RealMatrix X, RealMatrix Y) {
		
		/*XNX = [X, -X];
		H_G = XNX' * XNX;
		D = repmat(diag(H_G), [1, n_y]);
		XNXTY = XNX' * Y;
	    A = (X' * X + lambda  * eye(p)) \ (X' * Y);*/
		
		RealMatrix XNX = horzcat(X, uminus(X));
		RealMatrix H_G = XNX.transpose().multiply(XNX);
		RealMatrix D = repmat(diag(H_G), 1, ny);
		RealMatrix XNXTY = XNX.transpose().multiply(Y);
		RealMatrix A = mldivide(
						   plus(X.transpose().multiply(X), times(lambda, eye(p))), 
						   X.transpose().multiply(Y)
						   );
		
		/*AA = [subplus(A); subplus(-A)];
		C = -XNXTY + lambda;
		Grad = C + H_G * AA;
		tol = epsilon * norm(Grad);
		PGrad = zeros(size(Grad));*/
		
		RealMatrix AA = vertcat(subplus(A), subplus(uminus(A)));
		RealMatrix C = plus(uminus(XNXTY), lambda);
		RealMatrix Grad = plus(C, mtimes(H_G, AA));
		double tol = epsilon * norm(Grad);
		RealMatrix PGrad = zeros(size(Grad));
		
		ArrayList<Double> J = new ArrayList<Double>();
		double fval = 0;
		// J(1) = sum(sum((Y - X * A).^2)) / 2 + lambda * sum(sum(abs(A)));
		if (calc_OV) {
			fval = sum(sum(power(minus(Y, mtimes(X, A)), 2))).getEntry(0, 0) / 2 +
				   lambda * sum(sum(abs(A))).getEntry(0, 0);
			J.add(fval);
		}
		
		RealMatrix I_k = null;
		RealMatrix I_k_com = null;
		double d = 0;
		RealMatrix tmp = null;
		int k = 0;
		while (true) {	
			
			/*I_k = Grad < 0 | AA > 0;
		    I_k_com = not(I_k);
		    PGrad(I_k) = Grad(I_k);
		    PGrad(I_k_com) = 0;*/
			
			I_k = or(lt(Grad, 0), gt(AA, 0));
			I_k_com = not(I_k);
			PGrad = Grad.copy();
			logicalIndexingAssignment(PGrad, I_k_com, 0);
			
			d = norm(PGrad, inf);
		    if (d < tol) {
		        if (verbose)
		            System.out.println("Converge successfully!");
		        
		        break;
		    }
		    
		    /*for i = 1:2*p
		            AA(i, :) = max(AA(i, :) - (C(i, :) + H_G(i, :) * AA) ./ (D(i, :)), 0);
		    end
		    A = AA(1:p,:) - AA(p+1:end,:);*/
		    
		    for (int i = 0; i < 2 * p; i++) {
		    	tmp = max(
		    			minus(
	    					AA.getRowMatrix(i), 
	    					rdivide(
    							plus(C.getRowMatrix(i), mtimes(H_G.getRowMatrix(i), AA)),
    						    D.getRowMatrix(i)
    						    )
		    				),
		    		    0.0
		    		    );
		    						    
		    	AA.setRowMatrix(i, tmp);
		    }
		    
		    Grad = plus(C, mtimes(H_G,AA));
		    
		    k = k + 1;
		    if (k > maxIter) {
		    	if (verbose)
		    		System.out.println("Maximal iterations");
		    	break;
		    }
		    
		    if (calc_OV) {
				fval = sum(sum(power(minus(Y, mtimes(XNX, AA)), 2))).getEntry(0, 0) / 2 +
						lambda * sum(sum(abs(AA))).getEntry(0, 0);
				J.add(fval);
			}
		    
		    if (k % 10 == 0 && verbose) {
		    	if (calc_OV)
		    		System.out.format("Iter %d - ||PGrad||: %f, ofv: %f\n", k, d, J.get(J.size() - 1));
		    	else
		    		System.out.format("Iter %d - ||PGrad||: %f\n", k, d);

		    }
			
		}
		
		A = minus(
	    		  AA.getSubMatrix(0, p - 1, 0, ny - 1),
	    		  AA.getSubMatrix(p, 2 * p - 1, 0, ny - 1)
	    		  );
		return A;
		
	}

}
