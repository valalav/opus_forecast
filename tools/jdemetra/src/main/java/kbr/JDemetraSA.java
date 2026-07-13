package kbr;

import jdplus.toolkit.base.api.timeseries.TsData;
import jdplus.toolkit.base.api.timeseries.TsPeriod;
import jdplus.toolkit.base.api.modelling.ComponentInformation;
import jdplus.sa.base.api.ComponentType;
import jdplus.sa.base.api.SeriesDecomposition;
import jdplus.tramoseats.base.api.tramoseats.TramoSeatsSpec;
import jdplus.tramoseats.base.core.tramoseats.TramoSeatsKernel;
import jdplus.tramoseats.base.core.tramoseats.TramoSeatsResults;
import jdplus.x13.base.api.x13.X13Spec;
import jdplus.x13.base.core.x13.X13Kernel;
import jdplus.x13.base.core.x13.X13Results;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDate;
import java.util.*;

/**
 * JDemetra+ v3 Seasonal Adjustment CLI
 * 
 * Usage: java -jar jdemetra_sa.jar [options]
 * 
 * Options:
 *   --method     tramo-seats | x13  (default: tramo-seats)
 *   --input      path to CSV file (date,value per line)
 *   --output     path to output JSON (default: stdout)
 *   --freq       12 (monthly, default) | 4 (quarterly)
 */
public class JDemetraSA {
    
    public static void main(String[] args) {
        try {
            String method = getArg(args, "--method", "tramo-seats");
            String inputFile = getArg(args, "--input", null);
            String outputFile = getArg(args, "--output", null);
            int freq = Integer.parseInt(getArg(args, "--freq", "12"));
            
            if (inputFile == null) {
                System.err.println("Error: --input is required");
                System.err.println("Usage: java -jar jdemetra_sa.jar --input data.csv [--method tramo-seats|x13] [--freq 12|4]");
                System.exit(1);
            }
            
            // Read CSV
            List<String[]> rows = readCSV(inputFile);
            if (rows.isEmpty()) {
                System.err.println("Error: input CSV is empty");
                System.exit(1);
            }
            
            // Parse dates and values
            int startYear = 0, startMonth = 0;
            double[] values = new double[rows.size()];
            
            for (int i = 0; i < rows.size(); i++) {
                String dateStr = rows.get(i)[0].trim();
                double val = Double.parseDouble(rows.get(i)[1].trim());
                values[i] = val;
                
                if (i == 0) {
                    LocalDate d = parseDate(dateStr);
                    startYear = d.getYear();
                    startMonth = d.getMonthValue();
                }
            }
            
            // Create TsData
            TsPeriod start;
            if (freq == 12) {
                start = TsPeriod.monthly(startYear, startMonth);
            } else {
                start = TsPeriod.quarterly(startYear, (startMonth - 1) / 3 + 1);
            }
            TsData data = TsData.ofInternal(start, values);
            
            // Run seasonal adjustment
            StringBuilder json = new StringBuilder();
            json.append("{\n");
            json.append("  \"method\": \"").append(method).append("\",\n");
            json.append("  \"n\": ").append(values.length).append(",\n");
            
            SeriesDecomposition finals;
            
            if ("tramo-seats".equals(method)) {
                TramoSeatsSpec spec = TramoSeatsSpec.RSAfull;
                TramoSeatsKernel kernel = TramoSeatsKernel.of(spec, null);
                TramoSeatsResults results = kernel.process(data, null);
                finals = results.getFinals();
            } else if ("x13".equals(method)) {
                X13Spec spec = X13Spec.RSA5;
                X13Kernel kernel = X13Kernel.of(spec, null);
                X13Results results = kernel.process(data, null);
                // Use X11 D-tables directly for X-13
                jdplus.x13.base.core.x11.X11Results x11 = results.getDecomposition();
                TsData sa = x11.getD11();      // SA series
                TsData trend = x11.getD12();   // Trend
                TsData seasonal = x11.getD10(); // Seasonal factors
                TsData irregular = x11.getD13(); // Irregular
                
                appendSeries(json, "seasadj", sa);
                json.append(",\n");
                appendSeries(json, "trend", trend);
                json.append(",\n");
                appendSeries(json, "seasonal", seasonal);
                json.append(",\n");
                appendSeries(json, "irregular", irregular);
                json.append(",\n");
                json.append("  \"status\": \"OK\"\n");
                json.append("}\n");
                
                String result = json.toString();
                if (outputFile != null) {
                    Files.writeString(Path.of(outputFile), result);
                    System.err.println("OK: Results written to " + outputFile);
                } else {
                    System.out.println(result);
                }
                return;
            } else {
                System.err.println("Unknown method: " + method);
                System.exit(1);
                return;
            }
            
            // Extract components from SeriesDecomposition
            TsData sa = finals.getSeries(ComponentType.SeasonallyAdjusted, ComponentInformation.Value);
            TsData trend = finals.getSeries(ComponentType.Trend, ComponentInformation.Value);
            TsData seasonal = finals.getSeries(ComponentType.Seasonal, ComponentInformation.Value);
            TsData irregular = finals.getSeries(ComponentType.Irregular, ComponentInformation.Value);
            
            appendSeries(json, "seasadj", sa);
            json.append(",\n");
            appendSeries(json, "trend", trend);
            json.append(",\n");
            appendSeries(json, "seasonal", seasonal);
            json.append(",\n");
            appendSeries(json, "irregular", irregular);
            json.append(",\n");
            json.append("  \"status\": \"OK\"\n");
            json.append("}\n");
            
            // Output
            String result = json.toString();
            if (outputFile != null) {
                Files.writeString(Path.of(outputFile), result);
                System.err.println("OK: Results written to " + outputFile);
            } else {
                System.out.println(result);
            }
            
        } catch (Exception e) {
            System.err.println("FATAL: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(2);
        }
    }
    
    private static void appendSeries(StringBuilder json, String name, TsData ts) {
        json.append("  \"").append(name).append("\": [");
        if (ts != null && ts.length() > 0) {
            double[] vals = ts.getValues().toArray();
            for (int i = 0; i < vals.length; i++) {
                if (i > 0) json.append(", ");
                json.append(String.format(java.util.Locale.US, "%.6f", vals[i]));
            }
        }
        json.append("]");
    }
    
    private static List<String[]> readCSV(String path) throws IOException {
        List<String[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(path), "UTF-8"))) {
            String line;
            boolean firstLine = true;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                if (firstLine && line.charAt(0) == '\uFEFF') line = line.substring(1);
                firstLine = false;
                if (line.toLowerCase().startsWith("date") || line.toLowerCase().startsWith("дата")) continue;
                String[] parts = line.split("[,;\\t]");
                if (parts.length >= 2) rows.add(parts);
            }
        }
        return rows;
    }
    
    private static LocalDate parseDate(String s) {
        s = s.trim().replace("\"", "");
        String[] formats = {"yyyy-MM-dd", "dd.MM.yyyy", "MM/dd/yyyy", "yyyy/MM/dd"};
        for (String fmt : formats) {
            try {
                return LocalDate.parse(s, java.time.format.DateTimeFormatter.ofPattern(fmt));
            } catch (Exception ignored) {}
        }
        try {
            String[] p = s.split("[-/.]");
            if (p.length == 2) {
                int y = Integer.parseInt(p[0]);
                int m = Integer.parseInt(p[1]);
                if (y < 100) y += 2000;
                return LocalDate.of(y, m, 1);
            }
        } catch (Exception ignored) {}
        throw new RuntimeException("Cannot parse date: " + s);
    }
    
    private static String getArg(String[] args, String key, String defaultVal) {
        for (int i = 0; i < args.length - 1; i++) {
            if (key.equals(args[i])) return args[i + 1];
        }
        return defaultVal;
    }
}
