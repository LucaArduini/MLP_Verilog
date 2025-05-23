`timescale 1ns / 1ps
// 1ns: unit� di tempo (Time Unit) ? tutte le durate (es. #10) saranno interpretate come 10 nanosecondi
// 1ps: precisione di tempo (Time Precision) ? la simulazione considera il tempo con una risoluzione di 1 picosecondo

module mac_tb;

    parameter A_WIDTH = 8;
    parameter B_WIDTH = 8;
    parameter ACC_WIDTH = 32;
    parameter PROD_WIDTH = A_WIDTH + B_WIDTH;

    reg clk;
    reg start;
    reg valid;
    reg signed [A_WIDTH-1:0] a;
    reg signed [B_WIDTH-1:0] b;

    wire signed [ACC_WIDTH-1:0] result;

    // Instantiate MAC
    MLP_mac #(
        .A_WIDTH(A_WIDTH),
        .B_WIDTH(B_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) uut (
        .clk(clk),
        .start(start),
        .valid(valid),
        .a(a),
        .b(b),
        .result(result)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;	// Periodo di clock di 10ns

    // Test vectors
    reg signed [A_WIDTH-1:0] a_vec [0:3];
    reg signed [B_WIDTH-1:0] b_vec [0:3];
    integer i;

    // LUCA: Variabili per il calcolo del risultato atteso
    reg signed [ACC_WIDTH-1:0] expected_acc;
    reg signed [PROD_WIDTH-1:0] current_product;
    reg signed [ACC_WIDTH-1:0] current_product_ext;
    integer errors = 0;

    initial begin
        $display("Starting MAC testbench...");
        $dumpfile("mac.vcd");
        $dumpvars(0, mac_tb);

        // Initialize test values
        a_vec[0] = 3;   b_vec[0] = 2;
        a_vec[1] = -1;  b_vec[1] = 5;
        a_vec[2] = 4;   b_vec[2] = -2;
        a_vec[3] = 1;   b_vec[3] = 10;

        a = 0;
        b = 0;
        start = 0;
        valid = 0;
        expected_acc = 0; 	// Inizializza l'accumulatore atteso

        #20; 			// Attendi un po' per la stabilizzazione iniziale




        // --- Test 1: Operazione di START ---
        $display("\nTest 1: Start operation");
	// First input with start = 1
        @(negedge clk); // Cambia gli input al negedge per essere stabili al posedge
        a = a_vec[0];
        b = b_vec[0];
        start = 1;
        valid = 0;

        // Calcola il prodotto atteso esteso in segno
        current_product = a_vec[0] * b_vec[0];
        current_product_ext = {{(ACC_WIDTH - PROD_WIDTH){current_product[PROD_WIDTH-1]}}, current_product};
        expected_acc = current_product_ext;
        $display("Time: %0t: Driving a=%d, b=%d, start=1. Expected product_ext=%d", $time, a, b, current_product_ext);

        @(posedge clk); // Il DUT campiona start, a, b e aggiorna acc
        #1; // Piccolo ritardo per la propagazione del risultato (buona pratica)

        // Verifica il risultato
        if (result !== expected_acc) begin
            $display("Error after start @ %0t: Expected %d, Got %d", $time, expected_acc, result);
            errors = errors + 1;
        end else begin
            $display("Pass after start @ %0t: Result = %d", $time, result);
        end

        @(negedge clk); // Prepara per il prossimo ciclo
        start = 0;      // Disattiva start dopo un ciclo

        // --- Test successivi: Operazioni di ACCUMULATE (VALID) ---
        for (i = 1; i < 4; i = i + 1) begin
            $display("\nTest %0d: Accumulate operation %0d", i+1, i);
            @(negedge clk);
            a = a_vec[i];
            b = b_vec[i];
            valid = 1; // start è già 0

            // Calcola il prodotto corrente e aggiorna l'accumulatore atteso
            current_product = a_vec[i] * b_vec[i];
            current_product_ext = {{(ACC_WIDTH - PROD_WIDTH){current_product[PROD_WIDTH-1]}}, current_product};
            expected_acc = expected_acc + current_product_ext;
            $display("Time: %0t: Driving a=%d, b=%d, valid=1. Expected current_product_ext=%d, Expected acc=%d", $time, a, b, current_product_ext, expected_acc);

            @(posedge clk); // Il DUT campiona valid, a, b e aggiorna acc
            #1;

            // Verifica il risultato
            if (result !== expected_acc) begin
                $display("Error during accumulation (i=%0d) @ %0t: Expected %d, Got %d", i, $time, expected_acc, result);
                errors = errors + 1;
            end else begin
                $display("Pass during accumulation (i=%0d) @ %0t: Result = %d", i, $time, result);
            end
        end

        @(negedge clk);
        valid = 0; // Disattiva valid dopo l'ultimo ciclo di accumulazione

        #20;
        $display("\nFinal accumulated result = %0d", result);
        $display("Expected final accumulated result = %0d", expected_acc);

        if (errors == 0) begin
            $display("Testbench PASSED!");
        end else begin
            $display("Testbench FAILED with %0d errors.", errors);
        end
        $finish;
    end

endmodule