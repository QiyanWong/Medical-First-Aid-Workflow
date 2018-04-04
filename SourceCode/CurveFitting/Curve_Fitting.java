/**
 * Created by qiyanwang on 6/23/17.
 */
import java.io.*;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.*;
import org.apache.commons.csv.*;

import Jama.Matrix;
import com.csvreader.CsvReader;
import com.csvreader.CsvWriter;

import static java.io.File.separator;
import static java.lang.Math.abs;
//import org.apache.commons.math3.linear.Array2DRowRealMatrix;
//import org.apache.commons.math3.linear.RealMatrix;

public class Curve_Fitting {

    public static double fit(double[] x_in,double[] t_in,int n ) {


        int count = 5;
        double[] x_buffer = new double[count];
        double[] t_buffer = new double[count];
        int j = 0;
        for(int i = n; i <n + 5;i++){

            t_buffer[j] = t_in[i];
            x_buffer[j] = x_in[i];
            j++;
        }
        double value = t_in[n+5];

        double[] x_array = new double[count];
        double[] t_array = new double[count];

        for(int arraycount=0; arraycount<count; arraycount++)
        {

            x_array[arraycount]= x_buffer[arraycount];
            t_array[arraycount]= t_buffer[arraycount];

        }

//        System.out.println(Arrays.toString(x_array));
//        System.out.println(Arrays.toString(t_array));

        //construction of phi matrix
        double[][] phi = new double[3][count];
        for( int matrixcol=0; matrixcol<count; matrixcol++){

            phi[0][matrixcol]=1;

        }
        for( int matrixcol=0; matrixcol<count; matrixcol++){

            phi[1][matrixcol]=x_array[matrixcol];

        }

        for( int matrixrow=2; matrixrow<3; matrixrow++){

            for(int matrixcol=0; matrixcol<count; matrixcol++){

                phi[matrixrow][matrixcol]=phi[matrixrow-1][matrixcol]*phi[0][matrixcol];

            }

        }

        double beta = 11.11;
        double alpha = 0.005;

        RealMatrix phi_mat = new Array2DRowRealMatrix(phi);
        //   RealMatrix x_mat = new Array2DRowRealMatrix(x_array);
        //   RealMatrix t_mat = new Array2DRowRealMatrix(t_array);
        double[][] alphaIdent = new double[3][3];

        for(int diag=0; diag<3; diag++){

            alphaIdent[diag][diag]=alpha;

        }
        double[] phiTemp = new double[3];
        double[][] phiSum = new double[3][3];
        RealMatrix phiSum_mat = new Array2DRowRealMatrix(phiSum);
        for(int isum=0; isum<count; isum++){

            for(int matrixrow=0; matrixrow<3; matrixrow++){

                phiTemp[matrixrow] = phi[matrixrow][isum];

            }

            RealMatrix phiTemp_mat = new Array2DRowRealMatrix(phiTemp);
            phiSum_mat=phiSum_mat.add(phiTemp_mat.multiply(phiTemp_mat.transpose()));

        }

        RealMatrix alpha_mat = new Array2DRowRealMatrix(alphaIdent);
        RealMatrix sInv_mat = alpha_mat.add(phiSum_mat.scalarMultiply(beta));
        double[][] sInv_array = new double[3][3];
        for(int srow=0; srow<3; srow++){

            for(int scol=0; scol<3; scol++){

                sInv_array[srow][scol]=sInv_mat.getEntry(srow, scol);
            }

        }
        Matrix SImat = new Matrix(sInv_array);
        Matrix Smat = SImat.inverse();
        double[][] S_array = Smat.getArray();

        RealMatrix s_mat = new Array2DRowRealMatrix(S_array);
        //System.out.println(phi_mat.getEntry(2, 1));
        double[] phiTempM = new double[3];
        double[] phiSumM = new double[3];
        RealMatrix phiSumM_mat = new Array2DRowRealMatrix(phiSumM);
        for(int isum=0; isum<count; isum++){

            for(int matrixrow=0; matrixrow<3; matrixrow++){

                phiTempM[matrixrow] = phi[matrixrow][isum];
            }

            RealMatrix phiTempM_mat = new Array2DRowRealMatrix(phiTempM);
            phiSumM_mat= phiSumM_mat.add(phiTempM_mat.scalarMultiply(t_array[isum]));

        }
        RealMatrix phiT_mat = phi_mat.transpose();
        RealMatrix mean_mat = phiT_mat.multiply(s_mat.multiply(phiSumM_mat)).scalarMultiply(beta);
        RealMatrix variance_mat = phiT_mat.multiply(s_mat.multiply(phi_mat)).scalarAdd(1/beta);
        double[][] mean_array = mean_mat.getData();
        double[][] variance_array = variance_mat.getData();

        return abs(value - mean_array[count-1][0]);
    }

    public static void main(String[] args) throws IOException {

        String path = "duration.csv";
        FileReader readin = new FileReader(path);
        int count = 0;

        CSVParser records = null;
        records = CSVFormat.EXCEL.parse(readin);

        double[] time = new double[9999];
        double[] activity = new double[9999];
        for (Iterator i = records.iterator(); i.hasNext(); ++count) {

            CSVRecord record = (CSVRecord) i.next();
            activity[count] = Double.valueOf(record.get(0));
            time[count] =Double.valueOf(record.get(2));

        }

        double[] actual = new double[700];
        int k = 0;
        for(int i = 6; i<3900; i+=6){
            actual[k]  = time[i] ;
            k = k+1;
        }

//        for(int i = 0; i < 3918; i++)
//            System.out.println(time[i]);
        double[] record =new double[654];
        double sum = 0;
        int j=0;
        for(int i = 0; i < 653; i=i+6) {

           double temp = fit(activity, time, i);
            record[j++] = temp;
            sum += temp;

        }

        System.out.println("Average: " + sum/654);
        System.out.println("StandardDiviation: "+TestStatistics.getStandardDiviation(record));
        writeCsv(record,actual);

    }

    public static void writeCsv(double[] record,double[] actual){
        try {

            String csvFilePath = "duration_reuslt.csv";
            CsvWriter wr =new CsvWriter(csvFilePath,',',Charset.forName("SJIS"));
            for(int i = 0; i < 654; i++) {
                String[] contents = {""+ record[i],""+actual[i]};
                wr.writeRecord(contents);

            }
            wr.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
