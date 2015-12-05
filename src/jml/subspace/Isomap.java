package jml.subspace;

import static jml.matlab.Matlab.disp;
import static jml.matlab.Matlab.eq;
import static jml.matlab.Matlab.eye;
import static jml.matlab.Matlab.logicalIndexingAssignment;
import static jml.matlab.Matlab.size;

import java.awt.Color;

import javax.swing.JFrame;

import jml.manifold.Manifold;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.math.plot.Plot2DPanel;
import org.math.plot.Plot3DPanel;

/***
 * Isomap.
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 29th, 2013
 */
public class Isomap extends DimensionalityReduction {

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
		RealMatrix R = Isomap.run(X, K, r);
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
		plot.addScatterPlot("Isomap", Color.RED, x, y);
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
		plot3D.addScatterPlot("Isomap", Color.RED, x, y, z);
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
	public Isomap(int r) {
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
	public Isomap(int r, int K) {
		super(r);
		this.K = K;
	}

	@Override
	public void run() {
		this.R = run(X, K, r);
	}
	
	/**
	 * Isomap (isometric feature mapping).
	 * 
	 * @param X a d x n data matrix
	 * 
	 * @param K number of nearest neighbors
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 * @return a r x n matrix which is the r dimensional 
	 *         representation of the given n data examples
	 * 
	 */
	public static RealMatrix run(RealMatrix X, int K, int r) {
		
		// Step 1: Construct neighborhood graph
		RealMatrix D = Manifold.adjacency(X, "nn", K, "euclidean");
		logicalIndexingAssignment(D, eq(D, 0), Double.POSITIVE_INFINITY);
		logicalIndexingAssignment(D, eye(size(D)), 0);
		
		// Step 2: Compute shortest paths
		int d = size(D, 1);
		for (int k = 0; k < d; k++) {
			for (int i = 0; i < d; i++) {
				for (int j = 0; j < d; j++) {
					D.setEntry(i, j, Math.min(D.getEntry(i,j), D.getEntry(i,k) + D.getEntry(k,j)));
				}
			}
		}
		
		// Construct r-dimensional embedding
		return MDS.run(D, r);
		
	}

}
