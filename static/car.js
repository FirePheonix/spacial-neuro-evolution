class Car {
    constructor(x, y, width, height, controlType, angle = 0, maxSpeed = 3, color = "blue") {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;

        this.speed = 0;
        this.acceleration = 0.2;
        this.maxSpeed = maxSpeed;
        this.friction = 0.05;
        this.angle = angle;
        this.damaged = false;
        this.finished = false;
        this.minDist = Infinity;

        this.fittness = 0;

        this.useBrain = controlType == "AI";

        if (controlType != "DUMMY") {
            this.sensor = new Sensor(this);
            // +2 inputs for GPS (Sin/Cos of angle to target)
            this.brain = new NeuralNetwork(
                [this.sensor.rayCount + 2, 10, 4]
            );
        }
        this.controls = new Controls(controlType);

        this.img = new Image();
        this.img.src = "car.png"

        this.mask = document.createElement("canvas");
        this.mask.width = width;
        this.mask.height = height;

        const maskCtx = this.mask.getContext("2d");
        this.img.onload = () => {
            maskCtx.fillStyle = color;
            maskCtx.rect(0, 0, this.width, this.height);
            maskCtx.fill();

            maskCtx.globalCompositeOperation = "destination-atop";
            maskCtx.drawImage(this.img, 0, 0, this.width, this.height);
        }
    }

    update(roadBorders, traffic, target = null) {
        if (!this.damaged && !this.finished) {
            this.#move();
            this.polygon = this.#createPolygon();
            this.damaged = this.#assessDamage(roadBorders, traffic);

            // Task-based fitness: progress towards target
            if (target) {
                const dist = Math.hypot(this.x - target.x, this.y - target.y);
                if (!this.minDist || dist < this.minDist) {
                    this.minDist = dist;
                }
                // Reward for getting closer (max possible dist is roughly world size ~2000)
                // Base fitness on how close they've ever gotten
                this.fittness = (3000 - this.minDist) + (this.speed * 10);

                // Check if reached target
                if (dist < 40) {
                    this.finished = true;
                    this.fittness += 5000; // Giant bonus for finishing
                }
            } else {
                // Fallback to old behavior if no target
                this.fittness += this.speed;
            }
        }
        if (this.sensor) {
            this.sensor.update(roadBorders, traffic);
            const offsets = this.sensor.readings.map(
                s => s == null ? 0 : 1 - s.offset
            );

            // ─── GPS INPUTS ───
            // Give the brain a compass pointing to the target
            if (target) {
                // 1. Calculate the vector to the target
                const tx = target.x - this.x;
                const ty = target.y - this.y;

                // 2. Calculate the car's current forward vector (based on #move logic)
                // x -= sin(a), y -= cos(a) -> Vector is (-sin(a), -cos(a))
                const cx = -Math.sin(this.angle);
                const cy = -Math.cos(this.angle);

                // 3. Calculate the angle between these two vectors
                // Dot product: A . B = |A||B|cos(theta)
                // Cross product (2D): A x B = |A||B|sin(theta)
                // atan2(sin, cos) gives the signed angle

                // Normalize target vector isn't strictly necessary for atan2 but good for safety
                const tMag = Math.hypot(tx, ty) || 1;
                const tx_n = tx / tMag;
                const ty_n = ty / tMag;

                const dot = cx * tx_n + cy * ty_n;
                const cross = cx * ty_n - cy * tx_n; // Z-component of cross product

                // 4. Input the relative angle
                // If car is pointing at target, cross=0, dot=1 -> angle=0
                // If target is 90 deg right, cross=1, dot=0 -> angle=PI/2
                const angleDiff = Math.atan2(cross, dot);

                offsets.push(Math.sin(angleDiff));
                offsets.push(Math.cos(angleDiff));
            } else {
                offsets.push(0);
                offsets.push(0);
            }

            const outputs = NeuralNetwork.feedForward(offsets, this.brain);

            if (this.useBrain) {
                this.controls.forward = outputs[0];
                this.controls.left = outputs[1];
                this.controls.right = outputs[2];
                this.controls.reverse = outputs[3];
            }
        }
    }

    #assessDamage(roadBorders, traffic) {
        for (let i = 0; i < roadBorders.length; i++) {
            if (polysIntersect(this.polygon, roadBorders[i])) {
                return true;
            }
        }
        for (let i = 0; i < traffic.length; i++) {
            if (polysIntersect(this.polygon, traffic[i].polygon)) {
                return true;
            }
        }
        return false;
    }

    #createPolygon() {
        const points = [];
        const rad = Math.hypot(this.width, this.height) / 2;
        const alpha = Math.atan2(this.width, this.height);
        points.push({
            x: this.x - Math.sin(this.angle - alpha) * rad,
            y: this.y - Math.cos(this.angle - alpha) * rad
        });
        points.push({
            x: this.x - Math.sin(this.angle + alpha) * rad,
            y: this.y - Math.cos(this.angle + alpha) * rad
        });
        points.push({
            x: this.x - Math.sin(Math.PI + this.angle - alpha) * rad,
            y: this.y - Math.cos(Math.PI + this.angle - alpha) * rad
        });
        points.push({
            x: this.x - Math.sin(Math.PI + this.angle + alpha) * rad,
            y: this.y - Math.cos(Math.PI + this.angle + alpha) * rad
        });
        return points;
    }

    #move() {
        if (this.controls.forward) {
            this.speed += this.acceleration;
        }
        if (this.controls.reverse) {
            this.speed -= this.acceleration;
        }

        if (this.speed > this.maxSpeed) {
            this.speed = this.maxSpeed;
        }
        if (this.speed < -this.maxSpeed / 2) {
            this.speed = -this.maxSpeed / 2;
        }

        if (this.speed > 0) {
            this.speed -= this.friction;
        }
        if (this.speed < 0) {
            this.speed += this.friction;
        }
        if (Math.abs(this.speed) < this.friction) {
            this.speed = 0;
        }

        if (this.speed != 0) {
            const flip = this.speed > 0 ? 1 : -1;
            if (this.controls.left) {
                this.angle += 0.05 * flip;
            }
            if (this.controls.right) {
                this.angle -= 0.05 * flip;
            }
        }

        this.x -= Math.sin(this.angle) * this.speed;
        this.y -= Math.cos(this.angle) * this.speed;
    }

    draw(ctx, drawSensor = false) {
        if (this.sensor && drawSensor) {
            this.sensor.draw(ctx);
        }

        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(-this.angle);

        // ─── DEBUG: Draw GPS Vector ───
        // If this car has a target (passed in update context, but we need it here?)
        // Actually, draw doesn't receive target.
        // We can cheat and access global 'target' if available, or just skip it.
        // Or store target in 'update'. 
        // For now, let's just rely on the user observing behavior.
        // Wait, if I want to debug, I should store 'this.targetLocation' in update.

        if (!this.damaged) {
            ctx.drawImage(this.mask,
                -this.width / 2,
                -this.height / 2,
                this.width,
                this.height);
            ctx.globalCompositeOperation = "multiply";
        }
        ctx.drawImage(this.img,
            -this.width / 2,
            -this.height / 2,
            this.width,
            this.height);

        // Draw ID or Brain fitness?
        // ctx.fillStyle = "white"; ctx.fillText(Math.floor(this.fittness), 0, 0);

        ctx.restore();

    }
}