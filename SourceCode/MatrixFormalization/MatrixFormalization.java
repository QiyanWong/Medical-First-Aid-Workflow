import java.io.*;
import java.util.*;

/**
 * Created by qiyanwang on 7/18/17.
 */
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import static jdk.nashorn.internal.runtime.regexp.joni.encoding.CharacterType.ASCII;

/**
     * Created by qiyanwang on 6/22/17.
     */
    public class MatrixFormalization {
        static void readIn(String path,int namerow, int actirow, int startrow, int timeRow, ArrayList<Integer> duration,ArrayList<Integer> timearr, ArrayList<String> casename,ArrayList<String> actiname) throws IOException {

            FileReader readin = new FileReader(path);
            int count = 0;
            String time;
            String start;
            String name;
            String acti;
            CSVParser records = null;
            records = CSVFormat.EXCEL.parse(readin);

            for (Iterator i = records.iterator(); i.hasNext(); ++count) {
                CSVRecord record = (CSVRecord) i.next();
                time = record.get(timeRow);
                start = record.get(startrow);
                name = record.get(namerow);
                acti = record.get(actirow);
                timearr.add(Integer.valueOf(time));
                duration.add(Integer.valueOf(start));
                casename.add(name);
                actiname.add(acti);

            }
        }

        public static void main(String args[]) throws IOException {
            int size = 400;
            int cut = 63;
            String path = "newDataWithDuration.csv";

            MatrixFormalization a = new MatrixFormalization();
            ArrayList<Integer> array = new ArrayList<>();
            ArrayList<Integer> timearr = new ArrayList<>();
            ArrayList<String> casename = new ArrayList<>();
            ArrayList<String> actiname = new ArrayList<>();
            readIn(path, 0,1,2,3, array, timearr,casename,actiname);
            String temp="";
            int count = FindMany(casename.get(0), casename) ;
            System.out.println(count);
            int helper = 0;
            int same = 0;
            String name ="";
            int row = 0;
//            for(int i = 0; i < casename.size(); i++){
//                System.out.println(casename.get(i));
//            }

            for(int i = 0; i < array.size()-1; i += count){
                ArrayList<String> aaa = new ArrayList<String>();
                count = FindMany(casename.get(i), casename);
//                System.out.println(""+ i +":" + count);
            for(int caseindx = i; caseindx < i + count; caseindx++) {

                int start = array.get(caseindx) / cut;
                int time = timearr.get(caseindx) / cut;
                if(time > 200) time = 200;
                String activ = actiname.get(caseindx);
                String nextact = "";
                if(caseindx < array.size()-1) nextact = actiname.get(caseindx + 1);
                String preact = "";
                if(caseindx > 1) preact = actiname.get(caseindx - 1);
                System.out.println(preact +"  "+ activ +"  " + nextact );


                if(activ.equals(preact)) {
                    same ++;
                    for(int j = helper; j < start; j++){
                        temp += "0,";
                    }
                    for (int k = start; k < time-1; k++) {
                        temp += "1,";
                    }
                    temp += "1";
                    if (activ.equals(nextact)) {
                        helper = time;
                        continue;
                    }
                    else {
                        helper = 0;
                        for (int l = time; l < size-1; l++) {
                            temp += "0,";
                        }
                        temp+= "0";

                        aaa.add(temp);
                        temp = "";
                    }
                }

                else if(activ.equals(nextact)){
                    for (int j = 0; j < start; j++) {
                        temp += "0,";
                    }
                    for (int k = start; k < time - 1; k++) {
                        temp += "1,";
                    }
                    temp += "1";
                    helper = time;
                    continue;

                }
                else {
                    helper = 0;

                    for (int j = 0; j < start; j++) {
                        temp += "0,";
                    }
                    for (int k = start; k < time-1; k++) {
                        temp += "1,";
                    }
                    temp += "1";

                    for (int l = time; l < size-1; l++) {
                        temp += "0,";
                    }
                    temp +="0";

                    aaa.add(temp);
                        temp = "";
                    }

                row++;
                name = casename.get(i);

            }
                int limit = 10 - count + same;
                for (int add = 0; add < size-1; add++) {
                    temp += "0,";
                }
                temp+="0";

                for (int addzero = 0; addzero < limit; addzero++) {
                    aaa.add(temp);
                }
                temp = "";
                same = 0;
                boolean isSuccess = false;
//                System.out.print(name);
                isSuccess = a.exportCsv(new File("/Users/qiyanwang/Desktop/last/"+"a" + name + "_1"+".csv"),aaa);

            }

        }

        public static int FindMany(String name, ArrayList<String> casename){
            String temp = name;
            int count = 0;
            for(int i = 0; i < casename.size()-1; i++){
                if(casename.get(i).equals(name)){
                    for(int j = i; j < casename.size()-1; j++){
                        count++;
                        if(!casename.get(j).equals(name)){ break;}
                    }
                    break;
                }

            }
            return count - 1;
        }



        public static boolean exportCsv(File file, List<String> dataList) {
            boolean isSucess = false;

            FileOutputStream out = null;
            OutputStreamWriter osw = null;
            BufferedWriter bw = null;
            try {
                out = new FileOutputStream(file);
                osw = new OutputStreamWriter(out);
                bw = new BufferedWriter(osw);

                if (dataList != null && !dataList.isEmpty()) {
                    for (String data : dataList) {
                        bw.append(data).append("\r");
                    }
                }
                isSucess = true;
            } catch (Exception e) {
                isSucess = false;
            } finally {
                if (bw != null) {
                    try {
                        bw.close();
                        bw = null;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                if (osw != null) {
                    try {
                        osw.close();
                        osw = null;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                if (out != null) {
                    try {
                        out.close();
                        out = null;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }

            return isSucess;
        }










            public void exportCsv(ArrayList<String> array) {

                MatrixFormalization a = new MatrixFormalization();
                boolean isSuccess = a.exportCsv(new File("test.csv"),array );
                System.out.println(isSuccess);
            }

//        public String[] topFrequent(String[] combo, int k){
//            if(combo.length == 0){
//                return new String[0];
//            }
//            Map<String,Integer>freqMap = getFreqMap(combo);
//            PriorityQueue<Map.Entry<String, Integer>>(){
//                @Override
//
//            }
//        }
    }

