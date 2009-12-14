
function Field = Create_Field_From_States(phi, x)

    FieldLayer = zeros(size(phi,1),size(phi,2),size(phi,2));
    
    for FieldBasisIndex=1:size(phi,1) 
        FieldLayer(FieldBasisIndex,:,:) = x(FieldBasisIndex)*phi(FieldBasisIndex,:,:);
    end

    Field = sum(FieldLayer,1);