function h2l_s = srr_h2l_from_header(h_lr, h_hr, opt)
% function h2l_s = srr_h2l_from_header(h_lr, h_hr, opt)
%
% create h2l from nii fn given a reference header
%
% Same operator as the original polyshape version (still the one in
% filip-szczepankiewicz/Vis_NIMG_2021), but without polyshape:
%   - the overlap area of the hr and lr voxel (both convex parallelograms) is
%     computed with Sutherland-Hodgman clipping + the shoelace formula instead
%     of area(intersect(polyshape, polyshape))
%   - candidates whose bounding boxes do not overlap are skipped (their overlap
%     area is 0, so they never enter h2l and do not change sum(fraction))
%   - the sparse matrix is built once from triplets instead of element by
%     element (sparse() sums duplicates, same as the += in the original)
% Everything else follows the original + srr_lr_ind_from_pos line by line:
% same hr/lr positions, same 5x5 candidate order, same out-of-FOV skip, same
% early stop once single(sum(fraction)) >= 0.999999.
%
% Checked against the polyshape version on an 8-view ROVER phantom (1.1 mm
% in-plane, 5.5 mm slice): identical sparsity pattern in every view, max
% |difference| 2.6e-12, SRR output rel. L2 difference 2.3e-12, ~40x faster.

if (nargin < 3), opt.present = 1; end %#ok<NASGU>

hr_p = geometerics_from_header(h_hr);
lr_p = geometerics_from_header(h_lr);

R_hr = hr_p.fovrot([1,3], [1,3]);
R_lr = lr_p.fovrot([1,3], [1,3]);

% corner offsets, same as hr_pos + fovrot * cornervecs(k,:)'
hr_corner_off = R_hr * hr_p.cornervecs';   % 2 x 4
lr_corner_off = R_lr * lr_p.cornervecs';   % 2 x 4

% candidate order of srr_lr_ind_from_pos: meshgrid + reshape(cat(2,A',B'),[],2)
% -> i offset varies fastest, j offset slowest
d_vec = [0 -1 1 -2 2];
[di_c, dj_c] = ndgrid(d_vec, d_vec);
di_c = di_c(:);
dj_c = dj_c(:);
n_cand = numel(di_c);

n_hr = prod(hr_p.size);
max_entries = n_hr * 8;
rows = zeros(max_entries, 1);
cols = zeros(max_entries, 1);
vals = zeros(max_entries, 1);
n_entries = 0;

for i = 1:hr_p.size(1)
    for j = 1:hr_p.size(2)
        hr_ind = sub2ind(hr_p.size, i, j);

        hr_pos = R_hr * ...
            ([(i-1);(j-1)] .* [hr_p.pixdim_x;hr_p.qfac*hr_p.pixdim_z]) + ...
            [hr_p.qoffset_x; hr_p.qoffset_z];

        lr_ind_init = (R_lr \ (hr_pos - ...
                       [lr_p.qoffset_x ; lr_p.qoffset_z])) ./ ...
                       [lr_p.pixdim_x; lr_p.qfac*lr_p.pixdim_z];
        i_init = floor(lr_ind_init(1)) + 1;
        j_init = floor(lr_ind_init(2)) + 1;

        hr_poly = hr_pos + hr_corner_off;   % 2 x 4
        hr_min = min(hr_poly, [], 2);
        hr_max = max(hr_poly, [], 2);

        frac_sum = 0;
        c = 1;
        while single(frac_sum) < 0.999999 && c < n_cand + 1
            li = i_init + di_c(c);
            lj = j_init + dj_c(c);
            c = c + 1;

            if (li < 1 || li > lr_p.size(1)); continue; end
            if (lj < 1 || lj > lr_p.size(2)); continue; end

            lr_pos = R_lr * ([(li-1);(lj-1)] .* ...
                [lr_p.pixdim_x ; lr_p.qfac*lr_p.pixdim_z]) + ...
                [lr_p.qoffset_x ; lr_p.qoffset_z];
            lr_poly = lr_pos + lr_corner_off;

            lr_min = min(lr_poly, [], 2);
            lr_max = max(lr_poly, [], 2);
            if any(lr_max <= hr_min) || any(hr_max <= lr_min); continue; end

            fraction = convex_overlap_area(hr_poly, lr_poly) / hr_p.area;
            frac_sum = frac_sum + fraction;

            if fraction ~= 0
                n_entries = n_entries + 1;
                rows(n_entries) = sub2ind(lr_p.size, li, lj);
                cols(n_entries) = hr_ind;
                vals(n_entries) = fraction;
            end
        end
    end
end

h2l_s = sparse(rows(1:n_entries), cols(1:n_entries), vals(1:n_entries), ...
    prod(lr_p.size), n_hr);

end


function a = convex_overlap_area(subj, clip)
% area of the intersection of two convex polygons given as 2 x n vertex lists
% (any orientation): Sutherland-Hodgman clipping of subj by clip, then shoelace

    % make the clip polygon counter-clockwise so "inside" is left of each edge
    if signed_area(clip) < 0
        clip = clip(:, end:-1:1);
    end

    out = subj;
    n_clip = size(clip, 2);
    for e = 1:n_clip
        if isempty(out); break; end
        a1 = clip(:, e);
        a2 = clip(:, mod(e, n_clip) + 1);
        ex = a2(1) - a1(1);
        ey = a2(2) - a1(2);

        inp = out;
        n_in = size(inp, 2);
        out = zeros(2, 2 * n_in);
        n_out = 0;
        for k = 1:n_in
            p = inp(:, k);
            q = inp(:, mod(k, n_in) + 1);
            sp = ex * (p(2) - a1(2)) - ey * (p(1) - a1(1));
            sq = ex * (q(2) - a1(2)) - ey * (q(1) - a1(1));
            if sp >= 0
                n_out = n_out + 1;
                out(:, n_out) = p;
            end
            if (sp >= 0) ~= (sq >= 0)
                t = sp / (sp - sq);
                n_out = n_out + 1;
                out(:, n_out) = p + t * (q - p);
            end
        end
        out = out(:, 1:n_out);
    end

    if size(out, 2) < 3
        a = 0;
    else
        a = abs(signed_area(out));
    end
end


function s = signed_area(P)
    x = P(1, :);
    y = P(2, :);
    s = 0.5 * sum(x .* circshift(y, -1) - circshift(x, -1) .* y);
end


function p = geometerics_from_header(h)
% copy of the local function in srr_h2l_from_header.m

p.size  = double([h.dim(2) h.dim(4)]);

b = h.quatern_b; c = h.quatern_c; d = h.quatern_d;
a = sqrt(1.0-(b*b+c*c+d*d));

% code added by JJ
if a<10e-6
    a=0; %set a to 0 if very small (due to imprecision)
end
p.qfac = h.pixdim(1); % read qfac for proper Z axis

% end JJ

p.fovrot = double(quat2rotm([a b c d]));

p.pixdim_x = double(h.pixdim(2));
p.pixdim_z = double(h.pixdim(4));
p.area =  double(p.pixdim_x * p.pixdim_z);

p.qoffset_x = double(h.qoffset_x);
p.qoffset_z = double(h.qoffset_z);

p.cornervecs = double([p.pixdim_x/2 p.pixdim_z/2;
    -p.pixdim_x/2 p.pixdim_z/2;
    -p.pixdim_x/2 -p.pixdim_z/2;
    p.pixdim_x/2 -p.pixdim_z/2]);
end
