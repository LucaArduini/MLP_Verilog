`timescale 1ns/1ps

module tb_MLP_weight_mem;

    // Parameters for the testbench - should match DUT or be configurable
    localparam ADDR_WIDTH = 6;
    localparam DATA_WIDTH = 32;
    localparam CLK_PERIOD = 10; // Clock period in ns
    localparam MAX_ADDR = (1 << ADDR_WIDTH) - 1;

    // NB: ADDR_WIDTH e DATA_WIDTH qui hanno valori utili solo ai fini di test.
    // Nel nostro caso studio, i valori sono $clog2(2+1) e 16 rispettivamente.

    // Testbench signals
    reg                     clk;
    reg                     rst;
    reg  [ADDR_WIDTH-1:0]   addr;
    reg                     wr_en;
    reg  [DATA_WIDTH-1:0]   wr_data;
    wire [DATA_WIDTH-1:0]   rd_data;

    // Instantiate the Device Under Test (DUT)
    MLP_weight_mem #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .addr(addr),
        .wr_en(wr_en),
        .wr_data(wr_data),
        .rd_data(rd_data)
    );

    // Clock generator
    always begin
        #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test sequence
    initial begin
        $display("Starting Testbench for MLP_weight_mem");
        clk = 0;
        rst = 1;
        addr = 0;
        wr_en = 0;
        wr_data = 0;

        // Apply reset
        #(2*CLK_PERIOD);
        rst = 0;
        $display("[%0t] Reset de-asserted.", $time);
        @(posedge clk); // Wait for one clock edge after reset

        // --- Test Case 1: Write to address 0 and read back ---
        $display("[%0t] Test Case 1: Write to address 0 and read back.", $time);
        addr <= 0;
        wr_data <= 32'hAAAAAAAA;
        wr_en <= 1;
        @(posedge clk); // Write occurs on this edge
        
        // De-assert write enable for read. Address is already set.
        // rd_data will reflect the newly written data due to combinational read
        wr_en <= 0; 
        @(posedge clk); // Settle read (though combinational read means it's already valid)

        if (rd_data === 32'hAAAAAAAA) begin
            $display("[%0t] PASS: Read from addr 0 successful. Data: %h", $time, rd_data);
        end else begin
            $error("[%0t] FAIL: Read from addr 0. Expected %h, Got %h", $time, 32'hAAAAAAAA, rd_data);
        end
        
        // --- Test Case 2: Write to address 5 and read back ---
        $display("[%0t] Test Case 2: Write to address 5 and read back.", $time);
        addr <= 5;
        wr_data <= 32'hBEEFBEEF;
        wr_en <= 1;
        @(posedge clk);
        
        wr_en <= 0;
        @(posedge clk);

        if (rd_data === 32'hBEEFBEEF) begin
            $display("[%0t] PASS: Read from addr 5 successful. Data: %h", $time, rd_data);
        end else begin
            $error("[%0t] FAIL: Read from addr 5. Expected %h, Got %h", $time, 32'hBEEFBEEF, rd_data);
        end

        // --- Test Case 3: Read from address 0 again (check data persistence) ---
        $display("[%0t] Test Case 3: Read from address 0 again.", $time);
        addr <= 0;
        // wr_en is already 0
        @(posedge clk);

        if (rd_data === 32'hAAAAAAAA) begin
            $display("[%0t] PASS: Read from addr 0 (second time) successful. Data: %h", $time, rd_data);
        end else begin
            $error("[%0t] FAIL: Read from addr 0 (second time). Expected %h, Got %h", $time, 32'hAAAAAAAA, rd_data);
        end

        // --- Test Case 4: Write to max address and read back ---
        $display("[%0t] Test Case 4: Write to max address (%0d) and read back.", $time, MAX_ADDR);
        addr <= MAX_ADDR;
        wr_data <= 32'hC0DEC0DE;
        wr_en <= 1;
        @(posedge clk);
        
        wr_en <= 0;
        @(posedge clk);

        if (rd_data === 32'hC0DEC0DE) begin
            $display("[%0t] PASS: Read from addr %0d successful. Data: %h", $time, MAX_ADDR, rd_data);
        end else begin
            $error("[%0t] FAIL: Read from addr %0d. Expected %h, Got %h", $time, MAX_ADDR, 32'hC0DEC0DE, rd_data);
        end

        // --- Test Case 5: Attempt write with wr_en = 0 ---
        $display("[%0t] Test Case 5: Attempt write to address 5 with wr_en=0.", $time);
        // Address 5 currently holds 32'hBEEFBEEF
        addr <= 5;
        wr_data <= 32'h12345678; // Data we try to write (should fail)
        wr_en <= 0; // Write enable is OFF
        @(posedge clk);
        
        // Read back from address 5
        // Address is still 5, wr_en is still 0
        @(posedge clk); 

        if (rd_data === 32'hBEEFBEEF) begin
            $display("[%0t] PASS: Data at addr 5 unchanged when wr_en=0. Data: %h", $time, rd_data);
        end else begin
            $error("[%0t] FAIL: Data at addr 5 changed when wr_en=0. Expected %h, Got %h", $time, 32'hBEEFBEEF, rd_data);
        end

        // --- Test Case 6: Read from an unwritten address (optional - depends on RAM init) ---
        // BRAMs typically initialize to X or 0 in simulation if not explicitly initialized.
        // This DUT (MLP_weight_mem) does not explicitly initialize 'mem' on reset.
        // Therefore, in a typical Verilog simulation, uninitialized 'reg' elements
        // will default to 'X'. Some simulators might default to '0'.
        $display("[%0t] Test Case 6: Read from unwritten address (10).", $time);
        addr <= 10; // Assuming address 10 has not been written to
        wr_en <= 0;
        @(posedge clk);

        // Check for 'X' (all bits undefined)
        if (rd_data === {DATA_WIDTH{1'bx}}) begin
            $display("[%0t] INFO (TC6): Read from unwritten addr 10 is all 'X's. This is the expected behavior for uninitialized 'reg' in simulation. Data: %h", $time, rd_data);
        end
        // Check for '0' (all bits zero)
        else if (rd_data === {DATA_WIDTH{1'b0}}) begin
            $display("[%0t] INFO (TC6): Read from unwritten addr 10 is all '0's. This might be a simulator default for uninitialized BRAM-like structures or if the DUT implicitly initializes to zero. Data: %h", $time, rd_data);
        end
        // Check if any bit is 'X' or 'Z' (partially uninitialized)
        else if ($isunknown(rd_data)) begin
            $warning("[%0t] WARNING (TC6): Read from unwritten addr 10 contains 'X' or 'Z' bits, but is not all 'X's. This is unusual for a simple uninitialized memory read. Data: %h", $time, rd_data);
        end
        // Otherwise, it's some other defined value
        else begin
             $warning("[%0t] WARNING (TC6): Read from unwritten addr 10 yielded a defined, non-zero value. This is unexpected for an uninitialized memory location. Data: %h", $time, rd_data);
        end

        $display("[%0t] All tests completed.", $time);
        $finish;
    end

    // Optional: Monitor signals for debugging
    // initial begin
    //     $monitor("[%0t] clk=%b, rst=%b, addr=%h, wr_en=%b, wr_data=%h, rd_data=%h",
    //              $time, clk, rst, addr, wr_en, wr_data, rd_data);
    // end

endmodule