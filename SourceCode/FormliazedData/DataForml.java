import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Created by qiyanwang on 6/22/17.
 */
public class DataForml {
    static void readIn(String path, int actRow, int timeRow,HashMap<String, String> map) throws IOException {

        FileReader readin = new FileReader(path);
        int count = 0;
        String time;
        String activity;

        CSVParser records = null;
        records = CSVFormat.EXCEL.parse(readin);

        for (Iterator i = records.iterator(); i.hasNext(); ++count) {
            CSVRecord record = (CSVRecord) i.next();
            time = record.get(timeRow);
            activity = record.get(actRow);
            map.put(activity,time);
        }
    }


    public static void main(String args[]) throws IOException {
        String path = "start.csv";
        HashMap<String, String> map = new HashMap<String, String>();
        readIn(path, 1, 2, map);
        System.out.println(map.size());
        int count = 0;
        for(Map.Entry<String, String> entry : map.entrySet()) {
            count++;
            if(count <= 5) {

                System.out.println();
                System.out.println(entry.getKey() + ": " + entry.getValue());
            }
            else {
                System.out.println("--------------------------");
                System.out.println("label:  "+entry.getKey() + ": " +"value:  " + entry.getValue());
                count = 0;
                System.out.println("--------------------------");
            }

        }
    }
}
