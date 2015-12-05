package jml.subspace;

import java.awt.Color;

import javax.swing.JFrame;

import jml.matlab.Matlab;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.math.plot.Plot2DPanel;
import org.math.plot.Plot3DPanel;

import static jml.manifold.Manifold.*;
import static jml.matlab.Matlab.*;

/***
 * Locally Linear Embedding (LLE).
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 13th, 2013
 */
public class LLE extends DimensionalityReduction {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		double[][] data = {{0, 2, 3, 4}, {2, 0, 4, 5}, {3, 4.1, 5, 6}, {2, 7, 1, 6}};
		RealMatrix X = new BlockRealMatrix(data);
		/*int n = 20;
		int p = 10;
		X = rand(p, n);*/

		int K = 3;
		int r = 3;
		RealMatrix R = LLE.run(X, K, r);
		disp("Original Data:");
		disp(X);
		disp("Reduced Data:");
		disp(R);
		
		double[] x = R.getRow(0);
		double[] y = R.getRow(1);
		double[] z = R.getRow(2);
		
		// create your PlotPanel (you can use it as a JPanel)
		Plot2DPanel plot = new Plot2DPanel();

		// add a line plot to the PlotPanel
		// plot.addLinePlot("my plot", Color.RED, x, y);
		
		// add a scatter plot to the PlotPanel
		plot.addScatterPlot("LLE", Color.RED, x, y);
		plot.addLegend("North");
		
		/*plot.setAxisLabel(0, "this");
		plot.setAxisLabel(1, "that");*/
		// plot.addLabel("this", Color.RED, new double[]{0, 0});

		// put the PlotPanel in a JFrame, as a JPanel
		JFrame frame = new JFrame("A 2D Plot Panel");
		frame.setContentPane(plot);
		frame.setBounds(100, 100, 500, 500);
		frame.setVisible(true);
		
		Plot3DPanel plot3D = new Plot3DPanel();
		plot3D.addScatterPlot("LLE", Color.RED, x, y, z);
		plot3D.addLegend("North");
		
		/*plot.setAxisLabel(0, "this");
		plot.setAxisLabel(1, "that");*/
		// plot.addLabel("this", Color.RED, new double[]{0, 0});

		// put the PlotPanel in a JFrame, as a JPanel
		JFrame frame3D = new JFrame("A 3D Plot Panel");
		frame3D.setContentPane(plot3D);
		frame3D.setBounds(100, 100, 500, 500);
		frame3D.setVisible(true);
		
	}
	
	/**
	 * Number of nearest neighbors to construct the 
	 * neighborhood graph.
	 */        
	int K;
	
	/**
	 * Constructor.
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 */
	public LLE(int r) {
		super(r);
		
	}
	
	/**
	 * Constructor.
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 * @param K number of nearest neighbors to construct
	 *          the neighborhood graph
	 *          
	 */
	public LLE(int r, int K) {
		super(r);
		this.K = K;
	}

	@Override
	public void run() {
		this.R = run(X, K, r);
	}

	/**
	 * LLE (Locally Linear Embedding).
	 * 
	 * @param X a d x n data matrix
	 * 
	 * @param K number of nearest neighbors
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 * @return a r x n matrix which is the r dimensional 
	 *         representation of the given n examples
	 * 
	 */
	public static RealMatrix run(RealMatrix X, int K, int r) {
		String type = "nn";
		double param = K;
		RealMatrix A = adjacencyDirected(X, type, param, "euclidean");
		int N = size(X, 2);
		RealMatrix X_i = null;
		RealMatrix C_i = null;
		RealMatrix C = null;
		RealMatrix w = null;
		RealMatrix W = gt(A, 0);
		RealMatrix M = null;
		RealMatrix Ones = ones(K, 1);
	    RealMatrix I = eye(N);
	    int[] neighborIndices = null;
		for (int i = 0; i < N; i++) {
			neighborIndices = find(A.getRowVector(i));
			X_i = Matlab.getColumns(X, neighborIndices);
			C_i = X_i.subtract(repmat(getColumns(X, i), 1, K));
			// disp(C_i);
		    C = C_i.transpose().multiply(C_i);
		    C = C.add(diag(diag(C)));
		    w = mldivide(C, Ones);
		    w = rdivide(w, sumAll(w));
		    // disp(w);
		    setSubMatrix(W, new int[]{i}, neighborIndices, w);
		}
		// disp(W);
		M = I.subtract(W);
		M = M.transpose().multiply(M);
		// disp(M);
		RealMatrix U = eigs(M, r + 1, "sm")[0];
		/*disp(U);
		disp(eigs(M, r + 1, "sm")[1]);*/
		return times(Math.sqrt(N), getColumns(U, colon(1, r)).transpose());
		
	}

}
