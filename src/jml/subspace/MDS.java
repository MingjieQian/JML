package jml.subspace;

import java.awt.Color;

import javax.swing.JFrame;

import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.math.plot.Plot2DPanel;
import org.math.plot.Plot3DPanel;

import static jml.matlab.Matlab.*;

/***
 * Multi-dimensional Scaling (MDS).
 * 
 * @author Mingjie Qian
 * @version 1.0, Mar. 13th, 2013
 */
public class MDS extends DimensionalityReduction {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		double[][] data = {{0, 2, 3, 4}, {2, 0, 4, 5}, {3, 4.1, 5, 6}, {2, 7, 1, 6}};
		RealMatrix O = new BlockRealMatrix(data);
		/*int n = 20;
		int p = 10;
		O = rand(p, n);*/
		
		RealMatrix D = l2Distance(O, O);
		// fprintf("%g\n", norm(D.subtract(D.transpose())));
		RealMatrix R = MDS.run(D, 3);
		disp("Original Data:");
		disp(O);
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
		plot.addScatterPlot("MultiDimensional Scaling", Color.RED, x, y);
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
		plot3D.addScatterPlot("MultiDimensional Scaling", Color.RED, x, y, z);
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
	 * Constructor.
	 * 
	 * @param r number of dimensions to be reduced to
	 * 
	 */
	public MDS(int r) {
		super(r);
	}

	@Override
	public void run() {
		
	}
	
	/**
	 * Dimensionality reduction by MDS.
	 * 
	 * @param D an n x n dissimilarity matrix where n is the 
	 *          sample size
	 *          
	 * @param p number of dimensions to be reduced to
	 * 
	 * @return a p x n matrix which is the p dimensional 
	 *         representation of the given n objects with dissimilarity 
	 *         matrix D
	 *             
	 */
	public static RealMatrix run(RealMatrix D, int p) {
		
		if (norm(D.subtract(D.transpose())) > 0) {
			System.err.println("The dissimilarity matrix should be symmetric!");
			System.exit(1);
		}
		int n = D.getColumnDimension();
		RealMatrix A = times(-1d/2, times(D, D));
		RealMatrix H = eye(n).subtract(rdivide(ones(n), n));
		RealMatrix B = H.multiply(A).multiply(H);
		B = rdivide(plus(B, B.transpose()), 2);
		// fprintf("%g\n", norm(B.subtract(B.transpose())));
		RealMatrix[] eigRes = eigs(B, n, "lm");
		int k = 0;
		for (k = p - 1; k >= 0; k--) {
			if (eigRes[1].getEntry(k, k) > 0)
				break;
		}
		
		return getColumns(eigRes[0], 0, k).multiply(diag(power(diag(eigRes[1].getSubMatrix(0, k, 0, k)), 0.5))).transpose();
		
	}

}
