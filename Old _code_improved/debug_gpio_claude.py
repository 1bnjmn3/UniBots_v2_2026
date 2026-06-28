import lgpio
import time

h = lgpio.gpiochip_open(0)

# Set all three FL pins as outputs
lgpio.gpio_claim_output(h, 23)  # IN1 / forward
lgpio.gpio_claim_output(h, 27)  # IN2 / backward
lgpio.gpio_claim_output(h, 22)  # ENA / enable

# Spin forward: IN1=HIGH, IN2=LOW, ENA=HIGH
lgpio.gpio_write(h, 23, 1)
lgpio.gpio_write(h, 27, 0)
lgpio.gpio_write(h, 22, 1)

time.sleep(2)

# Stop
lgpio.gpio_write(h, 22, 0)
lgpio.gpio_close(h)

