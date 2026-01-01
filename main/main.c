/*
 * Project: Non-Invasive Diabetes Prediction (PPG Signal Acquisition)
 * Component: ESP32-S3 + MAX30102
 * Output Format: IR, RED (Raw Waveforms for Serial Plotter)
 * Update: SEPARATE LED CURRENTS (Red/IR)
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_log.h"
#include "esp_timer.h"

/* ======================================================================== */
/* USER CONFIGURATION SECTION                                               */
/* (Tune these parameters to optimize your signal)                          */
/* ======================================================================== */

// 1. I2C Speed (Hz)
//    Recommend 400000 (400kHz) for high sample rates.
#define I2C_SPEED_HZ        400000

// 2. LED Current (mA) - SEPARATED CHANNELS
//    Range: 0.0 to 51.0 mA.
//    Formula: Register Value = Current / 0.2
//    Example: 10.0mA -> 50 (0x32), 7.0mA -> 35 (0x23)
#define LED_CURRENT_RED_MA  10.0   // Red often needs less power (avoid skin saturation)
#define LED_CURRENT_IR_MA   7.0  // IR often needs more power (deeper penetration)

// 3. ADC Range (Sensitivity)
//    0 = 2048nA (Most Sensitive - Easy to saturate)
//    1 = 4096nA (Recommended start)
//    2 = 8192nA
//    3 = 16384nA (Least Sensitive - Hard to saturate)
#define ADC_RANGE_OPT       1

// 4. Sample Rate (Hz)
//    0 = 50Hz
//    1 = 100Hz (Recommended for 18-bit resolution)
//    2 = 200Hz
//    3 = 400Hz
//    4 = 800Hz  (Resolution drops to 17-bit or lower)
//    5 = 1000Hz
//    6 = 1600Hz
//    7 = 3200Hz
#define SAMPLE_RATE_OPT     3

// 5. Pulse Width (Integration Time)
//    0 = 69us  (15-bit resolution)
//    1 = 118us (16-bit resolution)
//    2 = 215us (17-bit resolution)
//    3 = 411us (18-bit resolution - Best for Glucose features)
#define PULSE_WIDTH_OPT     3

// 6. FIFO Rolling Average
//    0=1 sample (No averaging)
//    1=2 samples
//    2=4 samples (Recommended for smoothing)
//    3=8 samples
//    4=16 samples
//    5=32 samples
#define FIFO_AVG_OPT        0

/* ======================================================================== */
/* ======================================================================== */

/* --- Hardware Pin Configuration --- */
#define I2C_MASTER_SDA_IO          1     
#define I2C_MASTER_SCL_IO          2     
#define I2C_MASTER_NUM             I2C_NUM_0
#define MAX30102_ADDR              0x57

/* --- Algorithm Parameters --- */
#define FINGER_THRESHOLD           50000 

static const char *TAG = "PPG_TUNER";

/* ---------- I2C Helper Functions ---------- */
static esp_err_t max30102_write_reg(uint8_t reg, uint8_t val) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MAX30102_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_write_byte(cmd, val, true);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    return ret;
}

static esp_err_t max30102_read_reg(uint8_t reg, uint8_t *data, size_t len) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MAX30102_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MAX30102_ADDR << 1) | I2C_MASTER_READ, true);
    if (len > 1) {
        i2c_master_read(cmd, data, len - 1, I2C_MASTER_ACK);
    }
    i2c_master_read_byte(cmd, data + len - 1, I2C_MASTER_NACK);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
    return ret;
}

/* ---------- Initialization ---------- */
static void max30102_init_sensor(void) {
    // 1. Reset Sensor
    max30102_write_reg(0x09, 0x40); 
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // 2. Configure FIFO (Register 0x08)
    // Bit 7-5: SMP_AVE, Bit 4: FIFO_ROLLOVER_EN (1=Enable), Bit 3-0: FIFO_A_FULL
    uint8_t fifo_conf = (FIFO_AVG_OPT << 5) | 0x10; // 0x10 enables rollover
    max30102_write_reg(0x08, fifo_conf); 
    
    // 3. Configure Mode (Register 0x09)
    // Mode 0x03 = SpO2 (Red + IR)
    max30102_write_reg(0x09, 0x03); 
    
    // 4. Configure SpO2 Parameters (Register 0x0A)
    // Bit 6-5: ADC Range, Bit 4-2: Sample Rate, Bit 1-0: Pulse Width
    uint8_t spo2_conf = (ADC_RANGE_OPT << 5) | (SAMPLE_RATE_OPT << 2) | (PULSE_WIDTH_OPT);
    max30102_write_reg(0x0A, spo2_conf); 
    
    // 5. Configure LED Pulse Amplitude (Registers 0x0C & 0x0D)
    // Register 0x0C = LED1 (RED)
    // Register 0x0D = LED2 (IR)
    uint8_t led_pa_red = (uint8_t)(LED_CURRENT_RED_MA / 0.2);
    uint8_t led_pa_ir  = (uint8_t)(LED_CURRENT_IR_MA / 0.2);
    
    max30102_write_reg(0x0C, led_pa_red); // Red
    max30102_write_reg(0x0D, led_pa_ir);  // IR
    
    ESP_LOGI(TAG, "Sensor Configured:");
    ESP_LOGI(TAG, "- I2C Speed: %d Hz", I2C_SPEED_HZ);
    ESP_LOGI(TAG, "- LED RED Current: %.1f mA (Reg: 0x%02X)", LED_CURRENT_RED_MA, led_pa_red);
    ESP_LOGI(TAG, "- LED IR  Current: %.1f mA (Reg: 0x%02X)", LED_CURRENT_IR_MA, led_pa_ir);
    ESP_LOGI(TAG, "- SpO2 Config Reg (0x0A): 0x%02X", spo2_conf);
}

/* ---------- Main Task ---------- */
void ppg_task(void *arg) {
    uint8_t ptr_data[3];
    uint8_t fifo_data[6];
    
    while (1) {
        // Read FIFO Pointers
        if (max30102_read_reg(0x04, ptr_data, 3) == ESP_OK) {
            uint8_t write_ptr = ptr_data[0];
            uint8_t read_ptr = ptr_data[2];

            int samples_to_read = write_ptr - read_ptr;
            if (samples_to_read < 0) samples_to_read += 32;

            while (samples_to_read > 0) {
                if (max30102_read_reg(0x07, fifo_data, 6) == ESP_OK) {
                    
                    uint32_t red_val = ((fifo_data[0] << 16) | (fifo_data[1] << 8) | fifo_data[2]) & 0x03FFFF;
                    uint32_t ir_val  = ((fifo_data[3] << 16) | (fifo_data[4] << 8) | fifo_data[5]) & 0x03FFFF;

                    if (ir_val < FINGER_THRESHOLD) {
                        printf("0,0\n");
                    } else {
                        printf("%lu,%lu\n", ir_val, red_val);
                    }
                }
                samples_to_read--;
            }
        } else {
             ESP_LOGE(TAG, "I2C Error - Check Connections");
        }
        // Polling rate
        vTaskDelay(pdMS_TO_TICKS(10)); 
    }
}

void app_main(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_SPEED_HZ, // Uses User Config
    };
    ESP_ERROR_CHECK(i2c_param_config(I2C_MASTER_NUM, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0));
    
    max30102_init_sensor();
    xTaskCreate(ppg_task, "ppg_task", 4096, NULL, 5, NULL);
}