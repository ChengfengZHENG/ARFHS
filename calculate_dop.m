function dop = calculate_dop(data, target_pos)
    n = length(data.x);
    A = zeros(n, 3);
    for i = 1:n
        dx = data.x(i) - target_pos(1);
        dy = data.y(i) - target_pos(2);
        dz = data.z(i) - target_pos(3);
        dist = sqrt(dx^2 + dy^2 + dz^2);
        A(i, :) = [dx/dist, dy/dist, dz/dist];
    end
    H = A' * A;
    if rank(H) < 3
        dop = inf;
    else
        dop = sqrt(trace(inv(H)));
    end
end
